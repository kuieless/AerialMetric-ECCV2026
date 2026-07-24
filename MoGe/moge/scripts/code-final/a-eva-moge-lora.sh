#!/bin/bash

# ================= 1. 全局配置区 =================

# 实验名称 (Output 文件夹名字的一部分)
EXP_NAME="64r-2k"

# 权重设置 (修改这里来跑不同的 Checkpoint)
CHECKPOINT_STEP="00002000"
LORA_CONFIG="/home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json"
LORA_WEIGHT="/home/szq/moge2/MoGe/workspace/final-fintune2-130-lora-final6-64r/checkpoint/${CHECKPOINT_STEP}.pt"
LORA_RANK=64  # Allowed: 64, 96, 128. alpha is set to 2 * rank.
INTRINSICS_MODE="auto"  # auto, load, or none.

# 路径配置
DATA_ROOT="/data1/szq/Val"
SCRIPT_DIR="/home/szq/moge2/MoGe/moge/scripts/code-final" # 你的脚本所在的目录

# 输出路径定义 (自动生成)
# 最终结果会在: /data1/szq/Inference_Results_LoRA_Final_Run_00002000
INFER_OUT="/data1/szq/Inference_Results_${EXP_NAME}_${CHECKPOINT_STEP}"
EXTRACT_OUT="${INFER_OUT}_Extracted"

# 其他参数
SAMPLING_RATIO=1.0  # 1% 测试用，正式跑改 1.0
RESIZE_VAL=0         # 0 代表原图
BATCH_SIZE=4

# ================= 2. 执行流程 =================

set -e  # 遇到错误停止

echo "=========================================================="
echo "🚀 启动 LoRA 全流程 (Step: ${CHECKPOINT_STEP})"
echo "📂 数据源:   ${DATA_ROOT}"
echo "🔧 Config:   ${LORA_CONFIG}"
echo "⚖️  Weight:   ${LORA_WEIGHT}"
echo "🧩 LoRA:     rank=${LORA_RANK}, alpha=$((2 * LORA_RANK))"
echo "📷 Intrin.:  ${INTRINSICS_MODE}"
echo "📦 Batch:    ${BATCH_SIZE}"
echo "💾 输出:     ${INFER_OUT}"
echo "=========================================================="
echo ""

# --- 阶段 1: LoRA 推理 ---
echo ">>> [Step 1/3] Running Inference (LoRA)..."
# 注意：rank 必须与当前权重训练时使用的 LoRA rank 一致。
python "${SCRIPT_DIR}/a-infer_lora96-norm.py" \
    --input "${DATA_ROOT}" \
    --output "${INFER_OUT}" \
    --config "${LORA_CONFIG}" \
    --weight "${LORA_WEIGHT}" \
    --ratio "${SAMPLING_RATIO}" \
    --resize "${RESIZE_VAL}" \
    --batch_size "${BATCH_SIZE}" \
    --lora_rank "${LORA_RANK}" \
    --intrinsics_mode "${INTRINSICS_MODE}"

# --- 阶段 2: 结果抽取 ---
echo ""
echo ">>> [Step 2/3] Extracting Results..."

# 推理脚本会在 output 下创建 SourceName (Val)，所以输入是 ${INFER_OUT}/Val
EXTRACT_INPUT="${INFER_OUT}/Val"

python "${SCRIPT_DIR}/b-extra.py" \
    --input "${EXTRACT_INPUT}" \
    --output "${EXTRACT_OUT}" \
    --target "depth.npy" \
    --ext ".npy"

# --- 阶段 3: 结果评估 ---
echo ""
echo ">>> [Step 3/3] Running Evaluation..."

# 3.1 Oblique
if [ -d "${EXTRACT_OUT}/Oblique" ]; then
    echo "   🔎 Evaluating Oblique..."
    python "${SCRIPT_DIR}/c-eval-oblique.py" \
        --pred "${EXTRACT_OUT}/Oblique" \
        --gt "${DATA_ROOT}/Oblique"
fi

# 3.2 Bench
if [ -d "${EXTRACT_OUT}/Bench" ]; then
    echo "   🔎 Evaluating Bench..."
    python "${SCRIPT_DIR}/c-eval-bench.py" \
        --pred "${EXTRACT_OUT}/Bench" \
        --gt "${DATA_ROOT}/Bench"
fi

echo ""
echo "=========================================================="
echo "🎉 LoRA 流程结束! Checkpoint: ${CHECKPOINT_STEP}"
echo "📄 报告位置: ${EXTRACT_OUT}"
echo "=========================================================="
