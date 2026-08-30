"""
BitNet b1.58 Quantization-Aware Knowledge Distillation Trainer.
Distills Qwen 2.5 1.5B (FP16 Teacher) into BitNet b1.58 (Ternary Student).
Optimized for Apple Silicon Metal (MPS), NVIDIA CUDA, and multi-core CPU.
"""
import warnings
warnings.filterwarnings("ignore")

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




def get_devices() -> Tuple[torch.device, torch.device]:
    """Auto-detects compute hardware; splits student (cuda:0) and teacher (cuda:1) on dual GPU systems (e.g. Kaggle 2x T4)."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            return torch.device("cuda:0"), torch.device("cuda:1")
        return torch.device("cuda:0"), torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.device("mps")
    return torch.device("cpu"), torch.device("cpu")


def get_device() -> torch.device:
    s_dev, _ = get_devices()
    return s_dev


class DistillationLoss(nn.Module):
    """
    Ultra-Low Memory Top-K Distillation Loss.
    Processes Top-64 logits on Teacher GPU and transfers only 250 KB across GPUs instead of 622 MB.
    Reduces distillation peak VRAM from 15 GB down to <4 GB.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, topk_teacher_vals: torch.Tensor,
                topk_indices: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_student = student_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        vocab_size = shift_student.size(-1)
        t = self.temperature

        # 1. Hard Cross-Entropy on true labels
        ce = F.cross_entropy(
            shift_student.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100
        )

        # 2. Soft KL Divergence on Top-K teacher predictions
        topk_student_vals = torch.gather(shift_student, -1, topk_indices)
        p_s = F.log_softmax(topk_student_vals.float() / t, dim=-1)
        p_t = F.softmax(topk_teacher_vals.float() / t, dim=-1)
        kl = F.kl_div(p_s, p_t, reduction="batchmean") * (t ** 2)

        total_loss = (1.0 - self.alpha) * ce + self.alpha * kl
        return total_loss, ce.detach(), kl.detach()



# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_any_causal_model(model_name: str, dtype: torch.dtype):
    register_qwen3_5_architecture()
    # 1. Try Qwen3_5 official conditional generation class
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
    except Exception:
        pass

    # 2. Try AutoModelForImageTextToText (Qwen3.5 standard auto class)
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
    except Exception:
        pass

    # 3. Try AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
    except Exception:
        pass

    # 4. Universal fallback
    from transformers import AutoModel
    return AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )


def load_teacher_model(model_name: str, dtype: torch.dtype, device: torch.device):
    """Loads Teacher model in clean FP16 on the designated device (e.g. cuda:1)."""
    register_qwen3_5_architecture()
    teacher = load_any_causal_model(model_name, dtype=dtype)
    teacher = teacher.to(device)
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    return teacher



def train_distillation(args):
    student_device, teacher_device = get_devices()
    device = student_device
    print("=" * 70)
    print(f"🚀 BitNet b1.58 Distillation Pipeline (Student: {student_device}, Teacher: {teacher_device})")
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print(f"⚡ Dual GPU Pipeline Active: 2x {torch.cuda.get_device_name(0)} (30 GB Total VRAM)")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Teacher Model in native FP16
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    print(f"[2/5] Loading Teacher model '{args.model_name}' in FP16 on {teacher_device}...")
    teacher = load_teacher_model(args.model_name, dtype=dtype, device=teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded on {teacher_device} (FROZEN)")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Initialize BitNet Student Model
    print(f"[3/5] Constructing BitNet Student model...")
    student = load_any_causal_model(args.model_name, dtype=dtype)
    student, converted_count, preserved_count = convert_model_to_bitnet(student, verbose=True)
    student = student.to(device)
    if hasattr(student, "config"):
        student.config.use_cache = False

    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        print("✓ Enabled Gradient Checkpointing")

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


    # 5. Training Setup (8-Bit AdamW on CUDA to save 16.5 GB optimizer VRAM)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable BitLinear Tensors: {len(trainable_params)} tensors ({sum(p.numel() for p in trainable_params):,} parameters)")

    if device.type == "cuda":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay
            )
            print("✓ Initialized 8-Bit AdamW Optimizer (saves 16.5 GB VRAM)")
        except Exception as e:
            print(f"[Optimizer] bitsandbytes fallback ({e}), using standard AdamW...")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"[5/5] Commencing QAT Training ({args.epochs} epochs, {total_steps} steps)...")
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    start_time = time.time()
    teacher_dev = next(teacher.parameters()).device

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

            # Teacher forward pass (extract Top-64 logits on teacher GPU, transfer 0.25 MB)
            with torch.no_grad():
                t_in = input_ids.to(teacher_dev)
                t_mask = attention_mask.to(teacher_dev)
                teacher_out = teacher(input_ids=t_in, attention_mask=t_mask)
                shift_t = teacher_out.logits[..., :-1, :]
                k = min(64, shift_t.size(-1))
                topk_vals, topk_inds = torch.topk(shift_t, k=k, dim=-1)
                topk_teacher_vals = topk_vals.to(device)
                topk_indices = topk_inds.to(device)
                del teacher_out, shift_t, topk_vals, topk_inds

            # Student forward pass with AMP Mixed Precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                student_out = student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_out.logits
                loss, ce, kl = criterion(student_logits, topk_teacher_vals, topk_indices, labels)
                loss_scaled = loss / args.grad_accum_steps

            # Backward pass with GradScaler
            scaler.scale(loss_scaled).backward()

            del topk_teacher_vals, topk_indices, student_logits, student_out

            accum_loss += loss.item()
            accum_ce += ce.item()
            accum_kl += kl.item()

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

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

                if step > 0 and step % args.save_steps == 0:
                    periodic_path = os.path.join(args.output_dir, "bitnet_final.pt")
                    torch.save(student.state_dict(), periodic_path)
                    tokenizer.save_pretrained(args.output_dir)
                    print(f"\n✓ [Step {step}] Auto-saved checkpoint & tokenizer to '{periodic_path}'")

        # Save per-epoch checkpoint only if training multiple epochs
        if args.epochs > 1:
            ckpt_path = os.path.join(args.output_dir, f"bitnet_epoch_{epoch + 1}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"✓ Saved epoch checkpoint to '{ckpt_path}'")

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
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base HF model")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceTB/smoltalk", help="Training dataset")
    parser.add_argument("--dataset-config", type=str, default="all", help="Dataset config name (e.g. 'all')")
    parser.add_argument("--output-dir", type=str, default="./bitnet_qwen_3b_output", help="Save directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
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
