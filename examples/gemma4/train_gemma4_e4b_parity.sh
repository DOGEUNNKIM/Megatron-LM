#!/bin/bash
# Logit parity check: converted Megatron Gemma-4 E4B vs HF Gemma-4 E4B.
#
# Loads the converted Megatron checkpoint (TP=2) and the original HF model,
# runs the same token sequence through both, and checks that max |logit diff|
# is within --atol. Expected to pass with atol ~1.0 for bf16.
#
# Usage (from Megatron-LM root):
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/gemma4/train_gemma4_e4b_parity.sh
#
# Overrides:
#   GEMMA4_HF_DIR=...  GEMMA4_CKPT=...  ATOL=...  bash ...

set -euo pipefail

if [ ! -f "pretrain_gpt.py" ]; then
    echo "Error: run from the Megatron-LM root directory."
    exit 1
fi

GEMMA4_HF_DIR=${GEMMA4_HF_DIR:-$HOME/models/gemma-4-E4B-it}
GEMMA4_CKPT=${GEMMA4_CKPT:-$HOME/checkpoints/gemma4-e4b-megatron}
ATOL=${ATOL:-1.0}

if [ ! -d "$GEMMA4_HF_DIR" ]; then
    echo "Error: HF model dir not found: $GEMMA4_HF_DIR"
    echo "Set GEMMA4_HF_DIR=/path/to/gemma-4-E4B-it"
    exit 1
fi
if [ ! -f "$GEMMA4_CKPT/latest_checkpointed_iteration.txt" ]; then
    echo "Error: Megatron checkpoint not found at $GEMMA4_CKPT"
    echo "Set GEMMA4_CKPT=/path/to/gemma4-e4b-megatron"
    exit 1
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-6101}
TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR:-/tmp/gemma4_e4b_parity_logs}

export CUDA_DEVICE_MAX_CONNECTIONS=1
rm -rf "$TORCHRUN_LOG_DIR"
mkdir -p "$TORCHRUN_LOG_DIR"

echo "========================================"
echo "  Gemma-4 E4B parity check (TP=2)"
echo "  hf_dir : $GEMMA4_HF_DIR"
echo "  ckpt   : $GEMMA4_CKPT"
echo "  atol   : $ATOL"
echo "========================================"

torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes 1 --node_rank 0 \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    --log_dir "$TORCHRUN_LOG_DIR" \
    --redirects 3 --tee 3 \
    examples/gemma4/parity_check_e4b.py \
    --hf-dir "$GEMMA4_HF_DIR" \
    --megatron-ckpt "$GEMMA4_CKPT" \
    --atol "$ATOL"

echo "========================================"
echo "  Parity check PASSED"
echo "========================================"
