"""
BitNet b1.58 BitLinear Layer Implementation.
Supports:
1. Ternary weight quantization to {-1, 0, +1} with scale gamma.
2. 8-bit activation quantization with per-token/per-tensor scaling eta.
3. Straight-Through Estimator (STE) for backward pass gradients.
4. Packed 2-bit serialization (4 weights per byte).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def weight_quant(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes weights to ternary values {-1, 0, +1} using BitNet b1.58 scaling.
    gamma = mean(abs(w))
    w_quant = round(clip(w / gamma, -1, 1))
    """
    gamma = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / gamma
    w_clamped = torch.clamp(w_scaled, -1.0, 1.0)
    w_rounded = torch.round(w_clamped)

    # Straight-Through Estimator (STE)
    w_quant = w + (w_rounded * gamma - w).detach()
    return w_quant, gamma


def activation_quant(x: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes activations to INT8 [-128, 127] using absmax scaling.
    eta = max(abs(x))
    x_quant = round(clip(x * Q_b / eta, -Q_b, Q_b))
    """
    q_max = 2 ** (num_bits - 1) - 1  # 127 for 8-bit
    eta = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    scale = q_max / eta
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -q_max, q_max)
    x_rounded = torch.round(x_clamped)

    # STE
    x_quant = x + (x_rounded * (eta / q_max) - x).detach()
    return x_quant, eta


class BitLinear(nn.Linear):
    """
    BitNet b1.58 Linear Layer.
    Replaces standard nn.Linear layers in Transformer architectures.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 rms_norm_eps: float = 1e-5, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.rms_norm_eps = rms_norm_eps
        # LayerNorm / RMSNorm weight for input scaling
        self.layer_norm = nn.LayerNorm(in_features, eps=rms_norm_eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized activations and ternary weights.
        """
        # 1. Normalize input activations
        x_norm = self.layer_norm(x)

        # 2. Quantize activations to INT8
        x_quant, eta = activation_quant(x_norm)

        # 3. Quantize weights to ternary {-1, 0, 1}
        w_quant, gamma = weight_quant(self.weight)

        # 4. Compute linear projection
        return F.linear(x_quant, w_quant, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rms_norm_eps: float = 1e-5) -> "BitLinear":
        """Constructs a BitLinear layer from an existing PyTorch nn.Linear layer."""
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            rms_norm_eps=rms_norm_eps,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            bit_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                bit_linear.bias.copy_(linear.bias)
        return bit_linear

    def pack_ternary_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Packs ternary weights {-1, 0, 1} into 2-bit representation.
        Mapping:
          -1 -> 0b00 (0)
           0 -> 0b01 (1)
          +1 -> 0b10 (2)
        4 weights are packed into a single uint8 byte (4x compression).
        """
        with torch.no_grad():
            w_quant, gamma = weight_quant(self.weight)
            w_int = torch.round(w_quant / gamma).to(torch.int8) + 1  # maps {-1, 0, 1} to {0, 1, 2}
            w_flat = w_int.flatten()

            # Pad to multiple of 4
            pad_len = (4 - (len(w_flat) % 4)) % 4
            if pad_len > 0:
                w_flat = F.pad(w_flat, (0, pad_len), value=1)

            w_reshaped = w_flat.view(-1, 4)
            packed = (
                (w_reshaped[:, 0] & 0x03) |
                ((w_reshaped[:, 1] & 0x03) << 2) |
                ((w_reshaped[:, 2] & 0x03) << 4) |
                ((w_reshaped[:, 3] & 0x03) << 6)
            ).to(torch.uint8)

            return packed, gamma
