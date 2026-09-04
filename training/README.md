# Karma fine-tuning (English + Egyptian Arabic)

Fine-tunes **Qwen 2.5 0.5B** (or 1.5B) on Karma's bilingual dataset and exports a **`Q4_K_M` GGUF** that runs on the Pi.

## The data (`build_dataset.py` -> `dataset/`)

1426 samples, every user prompt unique. System prompts are byte-identical to the ones the robot uses at runtime (`config.PERSONA_SYSTEM_PROMPT`, `think._THINK_PROMPT`), so the model trains on exactly what it will see.

| File | N | Trains |
|---|---|---|
| `persona_en.jsonl` | 276 | Chill-friend voice, brevity, music taste, daily life |
| `persona_ar.jsonl` | 312 | Same, in Egyptian Arabic + local culture |
| `spontaneous_thoughts.jsonl` | 291 | think.py format: emotion/inflection JSON or `[silence]` (27% silence) |
| `vision_context.jsonl` | 222 | `Current Environment:` grounding, EN + AR scenes |
| `coding_mode.jsonl` | 110 | Beginner Python/JS, every reply carries a code block |
| `general_qa.jsonl` | 103 | Short factual answers, EN + AR |
| `boundary.jsonl` | 112 | Refusals in character (homework, legal/medical, hacking, self-harm) |

## Folder Structure

```
training/
├── karma_train_kaggle.ipynb        # All-in-one Kaggle Notebook (Train + GGUF Q4_K_M export)
├── generate_notebook.py            # Generator script for the Kaggle notebook
├── dataset/
│   ├── train.jsonl                 # Master combined dataset (1426 100% unique samples)
│   ├── persona_ar.jsonl            # Egyptian Arabic (عامية مصرية) banter & persona (312)
│   ├── persona_en.jsonl            # English chill friend banter & persona (276)
│   ├── spontaneous_thoughts.jsonl  # think.py JSON format + [silence] (291)
│   ├── vision_context.jsonl        # YOLO & object environment reactivity (222)
│   ├── coding_mode.jsonl           # Beginner-only code blocks + speech (110)
│   ├── general_qa.jsonl            # General reasoning and trivia anchors (103)
│   └── boundary.jsonl              # Safety refusals + identity, in-character (112)
├── build_dataset.py                # Dataset generator & extender
├── train_lora.py                   # Local LoRA fine-tuning & weight-merging script
└── README.md
```

---

## 1. Training on a free Kaggle GPU (recommended)

The notebook [`karma_train_kaggle.ipynb`](./karma_train_kaggle.ipynb) is completely self-contained:
1. Open [Kaggle](https://www.kaggle.com) -> Click **Create** -> **New Notebook**.
2. Go to **File** -> **Import Notebook** -> Upload `training/karma_train_kaggle.ipynb`.
3. In the right sidebar under **Settings**:
   - Turn **Internet** ON.
   - Set **Accelerator** to **GPU T4 x2** (or **GPU P100**).
4. Click **Run All**.
    - It compiles the 1426-sample dataset.
    - Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` (LoRA r=16, cosine, lr 1e-4, 3 epochs, 95/5 eval split) with loss on assistant tokens only.
    - Tests sample prompts in English and Egyptian Arabic.
    - Merges LoRA weights and compiles `llama.cpp`.
    - Directly outputs **`karma-qwen2.5-0.5b-q4_k_m.gguf`** (~350–470 MB depending on build) in `/kaggle/working`.
5. Download the GGUF file from Kaggle's **Output** panel and place it in Karma's `models/` folder:
   ```bash
   cp karma-qwen2.5-0.5b-q4_k_m.gguf models/model.gguf
   ```

---

## 2. Local Fine-Tuning (Apple Silicon Mac / Linux)

To train locally on your Mac with MPS or a Linux GPU:

```bash
# 1. Build or refresh dataset
python3 training/build_dataset.py

# 2. Run LoRA training
python3 training/train_lora.py
```

The script automatically saves the merged 16-bit model to `training/output/karma-merged/`.

`train_lora.py` works on old and new `trl`: loss masking uses a small collator vendored in the file (trl removed theirs in 0.20), and it picks `SFTConfig` or legacy `TrainingArguments`/`dataset_text_field` based on what's installed. `generate_notebook.py` writes the Kaggle notebook from these same sources, so edit the `.py` files and regenerate — never hand-edit the `.ipynb`.

---

## 3. Manual GGUF Conversion & Quantization

To convert a local merged model to GGUF using `llama.cpp`:

```bash
# 1. Convert to f16 GGUF
python3 llama.cpp/convert_hf_to_gguf.py training/output/karma-merged/ --outfile models/model-f16.gguf --outtype f16

# 2. Quantize to Q4_K_M for maximum speed on Raspberry Pi 4
./llama.cpp/build/bin/llama-quantize models/model-f16.gguf models/model.gguf Q4_K_M
rm models/model-f16.gguf
```
