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
    split: str = "train",
    max_samples: int = 50000,
    max_seq_len: int = 512,
    batch_size: int = 2,
    num_workers: int = 0
) -> DataLoader:
    """
    Downloads/streams dataset, tokenizes, packs sequences, and returns a PyTorch DataLoader.
    """
    print(f"[Dataset] Loading dataset '{dataset_name}' (subset: {max_samples} samples)...")

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"[Dataset] Note: Hugging Face datasets stream error ({e}), falling back to wikitext...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

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

    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
