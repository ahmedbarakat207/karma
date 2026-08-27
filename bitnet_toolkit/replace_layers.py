"""
Model Surgery Utility:
Recursively replaces standard PyTorch nn.Linear projection layers
in Transformer models with BitNet b1.58 BitLinear layers.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Set

from bitlinear import BitLinear


TARGET_LAYER_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2", "dense", "out_proj"
}

PRESERVE_LAYER_NAMES = {
    "embed_tokens", "wte", "wpe",
    "lm_head", "norm", "input_layernorm", "post_attention_layernorm", "visual"
}


def register_qwen3_5_architecture():
    """Dynamically registers Qwen 3.5 architectures with HuggingFace transformers."""
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        class Qwen3_5Config(Qwen3VLConfig):
            model_type = "qwen3_5"

        class Qwen3_5TextConfig(Qwen3VLTextConfig):
            model_type = "qwen3_5_text"

        class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
            config_class = Qwen3_5Config

        try:
            AutoConfig.register("qwen3_5", Qwen3_5Config)
            AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig)
            AutoModelForImageTextToText.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
            AutoModelForCausalLM.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
            AutoModel.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
        except Exception:
            pass
    except Exception:
        pass


register_qwen3_5_architecture()






def convert_model_to_bitnet(
    model: nn.Module,
    target_names: Set[str] = TARGET_LAYER_NAMES,
    preserve_names: Set[str] = PRESERVE_LAYER_NAMES,
    verbose: bool = True
) -> Tuple[nn.Module, int, int]:
    """
    Recursively traverses model hierarchy and swaps target nn.Linear layers for BitLinear.
    Returns:
        (converted_model, converted_count, preserved_count)
    """
    converted_count = 0
    preserved_count = 0

    def _replace_recursive(module: nn.Module, prefix: str = ""):
        nonlocal converted_count, preserved_count

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip explicitly preserved layers (e.g. embeddings, LM head)
            if any(p in name for p in preserve_names):
                preserved_count += 1
                continue

            if isinstance(child, nn.Linear):
                # Check if this linear layer matches targets
                if any(t in name for t in target_names) or len(target_names) == 0:
                    bit_layer = BitLinear.from_linear(child)
                    setattr(module, name, bit_layer)
                    converted_count += 1
                else:
                    preserved_count += 1
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)

    # Freeze preserved layers (embeddings & LM head) to eliminate 1.2 GB gradient allocation
    for name, param in model.named_parameters():
        if any(p in name for p in preserve_names):
            param.requires_grad = False

    if verbose:
        print(f"[BitNet Surgery] Successfully converted {converted_count} linear layers to BitLinear b1.58.")
        print(f"[BitNet Surgery] Preserved {preserved_count} sensitive layers (embeddings / lm_head frozen in full precision to save 1.2 GB VRAM).")

    return model, converted_count, preserved_count



def count_parameters(model: nn.Module) -> dict:
    """Returns detailed parameter statistics for full-precision vs ternary layers."""
    total_params = 0
    ternary_params = 0
    fp_params = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            ternary_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            fp_params += module.weight.numel()
        elif isinstance(module, nn.Embedding):
            fp_params += module.weight.numel()

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "total_params": total_params,
        "ternary_params": ternary_params,
        "fp_params": fp_params,
        "ternary_pct": (ternary_params / max(1, total_params)) * 100.0,
        "estimated_model_size_mb": (ternary_params * 1.58 / 8 + fp_params * 2) / (1024 * 1024)
    }
