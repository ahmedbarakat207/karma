"""
Dataset Loader & Tokenization Pipeline for BitNet Distillation.
Streams high-quality educational/conversational tokens with automatic chunk packing.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class TokenizedPackedDataset(Dataset):
    """
    Packs arbitrary text tokens into fixed sequence length chunks (e.g. 512 tokens).
    """

    def __init__(self, token_ids: List[int], max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        # Chunk tokens into fixed-length blocks
        num_chunks = len(token_ids) // max_seq_len
        self.chunks = []
        for i in range(num_chunks):
            start = i * max_seq_len
            self.chunks.append(token_ids[start:start + max_seq_len])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels
        }


def prepare_distillation_dataloader(
    tokenizer,
    dataset_name: str = "HuggingFaceTB/smoltalk",
    dataset_config: str = "all",
    split: str = "train",
    max_samples: int = 50000,
    max_seq_len: int = 512,
    batch_size: int = 2,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None
) -> DataLoader:
    """
    Downloads/streams dataset, tokenizes, packs sequences, and returns an optimized PyTorch DataLoader.
    """
    print(f"[Dataset] Loading dataset '{dataset_name}' (config: '{dataset_config}', max: {max_samples:,} samples)...")

    try:
        from datasets import load_dataset
        if "smoltalk" in dataset_name.lower():
            ds = load_dataset(dataset_name, dataset_config or "all", split=split, streaming=True)
        else:
            try:
                ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
            except Exception:
                ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"[Dataset] Note: Stream error ({e}), attempting fallback dataset...")
        from datasets import load_dataset
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        except Exception:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)


    all_tokens: List[int] = []
    count = 0

    for item in ds:
        if count >= max_samples:
            break
        text = ""
        if "messages" in item and isinstance(item["messages"], list):
            # Format ChatML conversation
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        elif "text" in item:
            text = item["text"].strip()

        if text:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
            count += 1

    print(f"[Dataset] Collected {len(all_tokens):,} total tokens across {count} samples.")

    packed_dataset = TokenizedPackedDataset(all_tokens, max_seq_len=max_seq_len)
    print(f"[Dataset] Packed into {len(packed_dataset)} sequence blocks (seq_len={max_seq_len}).")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    use_persistent = num_workers > 0

    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent
    )
