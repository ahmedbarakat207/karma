import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device="cpu")
out = model.generate("test")
print(type(out))
if isinstance(out, tuple):
    for i, x in enumerate(out): print(f"out[{i}]:", type(x))
elif isinstance(out, torch.Tensor):
    print("shape:", out.shape, "dtype:", out.dtype)
