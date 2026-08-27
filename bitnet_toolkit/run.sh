#!/usr/bin/env bash
# ==============================================================================
# BitNet b1.58 Production Upcycling Pipeline (50,000 Samples)
# Distills Qwen 2.5 1.5B into 1.58-Bit Ternary Model (~290 MB)
# Compatible with Apple Silicon Metal (MPS), NVIDIA CUDA, and multi-core CPU.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./model_output"
PACKED_DIR="./model_packed"
MODEL_NAME="${1:-${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}}"
DATASET_NAME="HuggingFaceTB/smoltalk"
MAX_SAMPLES=50000
MAX_SEQ_LEN=512
BATCH_SIZE=2
GRAD_ACCUM=16
EPOCHS=1
LR=1e-4
TEMP=2.0
ALPHA=0.5


echo "=============================================================================="
echo "🚀 Launching BitNet b1.58 Production Training (50,000 Samples / ~25M Tokens)"
echo "=============================================================================="
echo "Model:         $MODEL_NAME"
echo "Dataset:       $DATASET_NAME"
echo "Samples:       $MAX_SAMPLES"
echo "Sequence Len:  $MAX_SEQ_LEN"
echo "Effective BS:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "Output Dir:    $OUTPUT_DIR"
echo "Packed Dir:    $PACKED_DIR"
echo "=============================================================================="

# 1. Install / verify dependencies
echo -e "\n[Step 1/4] Checking dependencies..."
pip install -q -r requirements.txt

# 2. Run QAT Knowledge Distillation Training
echo -e "\n[Step 2/4] Starting QAT Distillation Training..."
python3 train_distill.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM \
    --max-seq-len $MAX_SEQ_LEN \
    --max-samples $MAX_SAMPLES \
    --lr $LR \
    --temperature $TEMP \
    --alpha $ALPHA \
    --gradient-checkpointing

# 3. Pack ternary weights into 2-bit binary format (~290 MB)
echo -e "\n[Step 3/4] Packing ternary weights into 2-bit format (~290 MB)..."
python3 export_bitnet.py \
    --checkpoint "$OUTPUT_DIR/bitnet_final.pt" \
    --model-name "$MODEL_NAME" \
    --output-dir "$PACKED_DIR"

# 4. Run Benchmark & Verification
echo -e "\n[Step 4/4] Running inference benchmark on upcycled model..."
python3 inference.py \
    --checkpoint "$OUTPUT_DIR/bitnet_final.pt" \
    --model-name "$MODEL_NAME"

echo -e "\n=============================================================================="
echo "🎉 Production Upcycling Complete!"
echo "Trained weights: $OUTPUT_DIR/bitnet_final.pt"
echo "Packed 2-bit:    $PACKED_DIR/bitnet_qwen_1.5b_packed_2bit.pt"
echo "=============================================================================="
