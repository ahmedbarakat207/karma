"""
BitNet b1.58 Quantization-Aware Knowledge Distillation Trainer.
Distills Qwen 2.5 1.5B (FP16 Teacher) into BitNet b1.58 (Ternary Student).
Optimized for Apple Silicon Metal (MPS), NVIDIA CUDA, and multi-core CPU.
"""
import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitlinear import BitLinear
from replace_layers import convert_model_to_bitnet, count_parameters, register_qwen3_5_architecture
from dataset_loader import prepare_distillation_dataloader




def get_device() -> torch.device:
    """Auto-detects best available compute hardware."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DistillationLoss(nn.Module):
    """
    Memory-efficient chunked Cross-Entropy and KL Divergence distillation.
    Prevents OOM on large vocabulary models (e.g. Qwen 248k vocab) by chunking sequence tokens.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, chunk_size: int = 64):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_student = student_logits[..., :-1, :]
        shift_teacher = teacher_logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        seq_len = shift_student.size(1)
        vocab_size = shift_student.size(-1)
        t = self.temperature

        total_ce = torch.tensor(0.0, device=student_logits.device)
        total_kl = torch.tensor(0.0, device=student_logits.device)
        total_loss = torch.tensor(0.0, device=student_logits.device)
        num_chunks = 0

        # Chunk along sequence dimension to keep transient activation memory tiny
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)

            s_chunk = shift_student[:, i:end_idx, :]
            t_chunk = shift_teacher[:, i:end_idx, :]
            l_chunk = shift_labels[:, i:end_idx]

            # 1. Chunked Cross-Entropy
            ce_chunk = self.ce_loss(s_chunk.reshape(-1, vocab_size), l_chunk.reshape(-1))

            # 2. Chunked KL Divergence
            p_s = F.log_softmax(s_chunk / t, dim=-1)
            p_t = F.softmax(t_chunk / t, dim=-1)
            kl_chunk = F.kl_div(p_s, p_t, reduction="batchmean") * (t ** 2)

            loss_chunk = (1.0 - self.alpha) * ce_chunk + self.alpha * kl_chunk

            total_loss = total_loss + loss_chunk
            total_ce = total_ce + ce_chunk.detach()
            total_kl = total_kl + kl_chunk.detach()
            num_chunks += 1

        total_loss = total_loss / max(1, num_chunks)
        total_ce = total_ce / max(1, num_chunks)
        total_kl = total_kl / max(1, num_chunks)

        return total_loss, total_ce, total_kl



# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_any_causal_model(model_name: str, dtype: torch.dtype):
    register_qwen3_5_architecture()
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        )
    except Exception:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True
            )
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
            except Exception:
                from transformers import AutoModel
                return AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )


def load_teacher_model(model_name: str, dtype: torch.dtype, device: torch.device):
    """Loads Teacher model with optional 4-bit NF4 quantization on CUDA to save 70% VRAM."""
    register_qwen3_5_architecture()

    if device.type == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            teacher = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ Loaded Teacher in 4-Bit NF4 (saves ~6 GB VRAM for Colab T4 GPU)")
            return teacher
        except Exception as e:
            print(f"[Note] 4-bit teacher fallback ({e}), attempting standard load...")

    return load_any_causal_model(model_name, dtype=dtype).to(device)


def train_distillation(args):
    device = get_device()
    print("=" * 70)
    print(f"🚀 BitNet b1.58 Distillation Pipeline (Target: {device.type.upper()})")
    print("=" * 70)

    # 0. Pre-initialize cuBLAS workspace before memory allocations
    if device.type == "cuda":
        try:
            _ = torch.zeros((1, 1), device=device) @ torch.zeros((1, 1), device=device)
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Teacher Model (with 4-bit VRAM compression on CUDA)
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    print(f"[2/5] Loading Teacher model '{args.model_name}'...")
    teacher = load_teacher_model(args.model_name, dtype=dtype, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded (FROZEN)")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Initialize BitNet Student Model
    print(f"[3/5] Constructing BitNet Student model...")
    student = load_any_causal_model(args.model_name, dtype=dtype)
    # Perform model surgery
    student, converted_count, preserved_count = convert_model_to_bitnet(student, verbose=True)
    student = student.to(device)



    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        print("✓ Enabled Gradient Checkpointing (60% VRAM reduction)")

    student.train()
    stats = count_parameters(student)
    print(f"Student Stats: {stats['ternary_params']:,} ternary weights ({stats['ternary_pct']:.1f}% of network)")
    print(f"Estimated 1.58-bit Model Size on Disk: ~{stats['estimated_model_size_mb']:.1f} MB")

    # 4. Prepare Distillation Dataset
    print(f"[4/5] Preparing training dataloader...")
    dataloader = prepare_distillation_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=getattr(args, "dataset_config", "all"),
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )


    # 5. Training Setup (8-Bit Paged AdamW on CUDA to save 16 GB optimizer memory)
    if device.type == "cuda":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(
                student.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay
            )
            print("✓ Initialized 8-Bit Paged AdamW Optimizer (saves 16 GB memory with GPU/CPU paging)")
        except Exception as e:
            print(f"[Optimizer] 8-bit Adam fallback ({e}), using standard AdamW...")
            optimizer = torch.optim.AdamW(
                student.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay
            )
    else:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )

    total_steps = (len(dataloader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=args.lr * 0.1
    )

    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)

    print(f"[5/5] Commencing QAT Training ({args.epochs} epochs, {total_steps} steps)...")
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kl = 0.0

        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits

            # Student forward pass (quantized weights + quantized activations)
            student_out = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits

            # Calculate distillation loss
            loss, ce, kl = criterion(student_logits, teacher_logits, labels)
            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()


            # Free transient forward tensors immediately
            del teacher_logits, teacher_out, student_logits, student_out

            accum_loss += loss.item()
            accum_ce += ce.item()
            accum_kl += kl.item()

            if (i + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                avg_loss = accum_loss / args.grad_accum_steps
                avg_ce = accum_ce / args.grad_accum_steps
                avg_kl = accum_kl / args.grad_accum_steps
                accum_loss = 0.0
                accum_ce = 0.0
                accum_kl = 0.0

                pbar.set_postfix({
                    "loss": f"{avg_loss:.3f}",
                    "ce": f"{avg_ce:.3f}",
                    "kl": f"{avg_kl:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # Save checkpoint after each epoch
        ckpt_path = os.path.join(args.output_dir, f"bitnet_epoch_{epoch + 1}.pt")
        torch.save(student.state_dict(), ckpt_path)
        print(f"✓ Saved checkpoint to '{ckpt_path}'")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 Training Completed in {total_time / 60:.1f} minutes!")
    final_path = os.path.join(args.output_dir, "bitnet_final.pt")
    torch.save(student.state_dict(), final_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved final BitNet model weights to '{final_path}'")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 Knowledge Distillation Trainer")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base HF model")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceTB/smoltalk", help="Training dataset")
    parser.add_argument("--dataset-config", type=str, default="all", help="Dataset config name (e.g. 'all')")
    parser.add_argument("--output-dir", type=str, default="./bitnet_qwen_1.5b_output", help="Save directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size (1 recommended for large models on 15GB GPUs)")
    parser.add_argument("--grad-accum-steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max training samples")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="KL vs CE loss blend ratio")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="LR warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable grad checkpointing")

    args = parser.parse_args()

    train_distillation(args)


if __name__ == "__main__":
    main()
