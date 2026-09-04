#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-0.5B (or 1.5B) with LoRA on the karma dataset.

Loss is computed on assistant tokens only, using a small collator
vendored here (trl removed theirs in 0.20), so this runs on both
old trl (<0.20) and new trl (SFTConfig).

    pip install torch transformers datasets peft trl accelerate sentencepiece
"""

import gc
import inspect
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "train.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "karma-lora")
MERGED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "karma-merged")

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def _find_sublist(haystack, needle):
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Masks prompt tokens with -100 so loss only covers assistant tokens."""

    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.resp_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            idx = _find_sublist(ids, self.resp_ids)
            if idx == -1:
                if not getattr(self, "_warned", False):
                    print("warning: response template missing in a sample, keeping full loss for it")
                    self._warned = True
                continue
            cutoff = idx + len(self.resp_ids)
            labels[i, :cutoff] = -100
        batch["labels"] = labels
        return batch


def train():
    if not os.path.exists(DATASET_PATH):
        print(f"dataset not found at {DATASET_PATH}, run build_dataset.py first")
        sys.exit(1)

    print(f"base model: {BASE_MODEL_NAME}")
    print(f"dataset:    {DATASET_PATH}")
    print(f"output:     {OUTPUT_DIR}")

    # tokenizer
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # masking assumes prompt comes first

    # dataset
    print("loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"loaded {len(dataset):,} samples")

    # render messages with the chat template
    def apply_chat_template(batch):
        formatted = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            formatted.append(text)
        return {"text": formatted}

    dataset = dataset.map(apply_chat_template, batched=True)
    missing = sum("<|im_start|>assistant" not in t for t in dataset["text"])
    assert missing == 0, f"{missing} samples missing the assistant header"
    print("assistant header present in all samples")

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"split: {len(train_dataset)} train / {len(eval_dataset)} eval")

    # base model
    print(f"loading base model {BASE_MODEL_NAME}...")
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if has_cuda else torch.float32)
    device_map = "auto" if has_cuda else {"": "cpu"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
    except TypeError:
        # transformers<4.46 uses torch_dtype instead of dtype
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
    model.config.use_cache = False

    # lora
    print("attaching LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # training args
    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # conservative for the 0.5B model
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        fp16=use_fp16,
        bf16=use_bf16,
        seed=42,
        report_to="none"
    )
    try:
        from trl import SFTConfig
        training_args = SFTConfig(max_length=1024, packing=False, **training_kwargs)
        print("using SFTConfig")
    except Exception as e:
        print(f"SFTConfig missing ({e}), falling back to TrainingArguments")
        training_args = TrainingArguments(**training_kwargs)

    # loss only on assistant replies, not on prompts
    collator = CompletionOnlyCollator(tokenizer=tokenizer, mlm=False)

    def _tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    from trl import SFTTrainer
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        tokenized_train = train_dataset.map(_tokenize_fn, batched=True, remove_columns=["messages", "text"])
        tokenized_eval = eval_dataset.map(_tokenize_fn, batched=True, remove_columns=["messages", "text"])
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=collator,
            processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            data_collator=collator,
            args=training_args
        )

    print("training...")
    trainer.train()

    print(f"saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # merge lora back into the base model for gguf export
    print(f"merging into {MERGED_DIR}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    del model
    del merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"done, merged model at {MERGED_DIR}")
    print("next: convert to gguf for the pi:")
    print("  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp")
    print("  pip install -r llama.cpp/requirements.txt")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outfile models/model-f16.gguf --outtype f16")
    print("  cmake -B llama.cpp/build -DLLAMA_CURL=OFF && cmake --build llama.cpp/build --config Release -t llama-quantize -j4")
    print("  ./llama.cpp/build/bin/llama-quantize models/model-f16.gguf models/model.gguf Q4_K_M")


if __name__ == "__main__":
    train()
