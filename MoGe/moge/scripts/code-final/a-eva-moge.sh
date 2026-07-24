

#!/bin/bash

# ================= 1. 全局配置区 =================

# 实验名称
EXP_NAME="wild-base"
# EXP_NAME="vitl-full-16.5k"

# 路径配置
DATA_ROOT="/data1/szq/Val"
SCRIPT_DIR="/home/szq/moge2/MoGe/moge/scripts/code-final"
# MODEL_PATH="/home/szq/moge2/MoGe/vitl-normal.pt"
# MODEL_PATH="//home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-2/checkpoint/00016500_ema.pt"
MODEL_PATH="/home/szq/moge2/MoGe/workspace/final-ground/checkpoint/00000600_ema.pt"

# 输出路径
INFER_OUT="/data1/szq/Inference_Results_${EXP_NAME}"
EXTRACT_OUT="${INFER_OUT}_Extracted"

# 参数
SAMPLING_RATIO=0.3
RESIZE_VAL=0

# ================= 2. 阶段控制逻辑 (核心修改) =================

# 默认从第 1 步开始
START_STEP=${1:-1}  # 接收第一个参数，如果没有传参，默认为 1

echo "=========================================================="
echo "🚀 启动自动化脚本 (从 Step ${START_STEP} 开始)"
echo "📂 数据源:      ${DATA_ROOT}"
echo "💾 推理输出:    ${INFER_OUT}"
echo "📂 抽取输出:    ${EXTRACT_OUT}"
echo "=========================================================="
echo ""

set -e # 遇到错误立即停止

# ================= Step 1: 推理 =================
if [ ${START_STEP} -le 1 ]; then
    echo ">>> [Step 1/3] Running Inference (Base Model)..."
    python "${SCRIPT_DIR}/a-infer.py" \
        --input "${DATA_ROOT}" \
        --output "${INFER_OUT}" \
        --model "${MODEL_PATH}" \
        --ratio "${SAMPLING_RATIO}" \
        --resize "${RESIZE_VAL}"
else
    echo ">>> [Step 1/3] Skipping Inference (User Requested)"
fi

# ================= Step 2: 抽取 =================
if [ ${START_STEP} -le 2 ]; then
    echo ""
    echo ">>> [Step 2/3] Extracting & Flattening Results..."
    
    # 自动推断输入目录
    EXTRACT_INPUT="${INFER_OUT}/Val"
    
    # 检查输入目录是否存在，防止跳过第一步但没有数据的情况
    if [ ! -d "${EXTRACT_INPUT}" ]; then
        echo "❌ Error: Step 1 结果不存在: ${EXTRACT_INPUT}"
        echo "   请先运行 Step 1，或检查路径是否正确。"
        exit 1
    fi

    python "${SCRIPT_DIR}/b-extra.py" \
        --input "${EXTRACT_INPUT}" \
        --output "${EXTRACT_OUT}" \
        --target "depth.npy" \
        --ext ".npy"
else
    echo ">>> [Step 2/3] Skipping Extraction (User Requested)"
fi

# ================= Step 3: 评估 =================
if [ ${START_STEP} -le 3 ]; then
    echo ""
    echo ">>> [Step 3/3] Running Evaluation..."
    
    # 3.1 评估 Oblique (使用你最新的 pixel 级跳过脚本)
    if [ -d "${EXTRACT_OUT}/Oblique" ]; then
        echo "   🔎 Evaluating Oblique Dataset..."
        # 注意：这里改成了 pixel 版本的脚本名，请确保文件名一致
        python "${SCRIPT_DIR}/c-eval-oblique.py" \
            --pred "${EXTRACT_OUT}/Oblique" \
            --gt "${DATA_ROOT}/Oblique"
    else
        echo "   ⚠️ Skip Oblique: Directory not found."
    fi

    # 3.2 评估 Bench
    if [ -d "${EXTRACT_OUT}/Bench" ]; then
        echo "   🔎 Evaluating Bench Dataset..."
        python "${SCRIPT_DIR}/c-eval-bench.py" \
            --pred "${EXTRACT_OUT}/Bench" \
            --gt "${DATA_ROOT}/Bench"
    else
        echo "   ⚠️ Skip Bench: Directory not found."
    fi

        # 3.3 评估 Wild 数据集 (绝对尺度模式)
    if [ -d "${EXTRACT_OUT}/Wild" ]; then
        echo "   🔎 Evaluating Wild Dataset (Metric Mode)..."
        # 这里调用新的脚本名 c-eval-wild-metric.py
        python "${SCRIPT_DIR}/c-eval-wild.py" \
            --pred "${EXTRACT_OUT}/Wild" \
            --gt "${DATA_ROOT}/Wild"
    else
        echo "   ⚠️ Skip Wild: Directory not found."
    fi
else
    echo ">>> [Step 3/3] Skipping Evaluation (User Requested)"
fi

echo ""
echo "=========================================================="
echo "🎉 流程结束!"
echo "=========================================================="