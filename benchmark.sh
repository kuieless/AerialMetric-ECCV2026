#!/bin/bash
# ============================================================================
# AerialMetric Benchmark Script (task queue + multi-GPU scheduler)
#
# Usage:
#   bash benchmark.sh 0,1,2,3    # 4 GPUs in parallel
#   bash benchmark.sh 0,1        # 2 GPUs in parallel
#   bash benchmark.sh 0          # single GPU
#
# Weights:
#   MoGe-2 ViT-Large (official): https://huggingface.co/Ruicheng/moge-2-vitl-normal
#   MoGe-2 Aerial LoRA:          https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026
# ============================================================================
set -euo pipefail

# ====== Config ======
CONDA_ENV="mogefresh3"
PROJECT_ROOT="/home/szq/moge2-fresh4"
OUTPUT_ROOT="/data1/szq/moge2-fresh4"
IFS=',' read -ra GPU_LIST <<< "${1:-0,1,2,3}"
N_GPUS=${#GPU_LIST[@]}

# ---- Model Weights ----
# MoGe-2 ViT-Large base checkpoint, download:
#   https://huggingface.co/Ruicheng/moge-2-vitl-normal
MOGE2_BASE="/home/szq/moge2-ed/vitl-normal.pt"

# MoGe-2 Aerial LoRA weights (paper checkpoint)
MOGE2_AERIAL="/data1/szq/moge2/权重/workspace/weights/Moge2-Aerial.pt"
LORA_CONFIG="$PROJECT_ROOT/MoGe/configs/Final_train/config-lora-all.json"

# ---- Datasets ----
# Standard layout (no per-sample intrinsics)
DECOUPLED="/data1/szq/Val/decoupled"
DECOUPLED_MASK="/data1/szq/Val/decoupled-masks"
OBLIQUE="/data1/szq/Val/Oblique"
OBLIQUE_MASK="/data1/szq/Val/Oblique-masks"
WILD="/data1/szq/Val/Wild"

# Norm-style layout (with meta.json intrinsics per sample)
DECOUPLED_NORM="/data1/szq/Val/decoupled-norm"
OBLIQUE_NORM="/data1/szq/Val/Oblique-norm"

# ---- Ground Benchmark ----
GROUND_DIR="$PROJECT_ROOT/Ground_MoGe"
GROUND_CFG="$GROUND_DIR/configs/eval/ground_metric_benchmarks_local.json"
AERIAL_CLI="$PROJECT_ROOT/MoGe/moge/scripts/code-final/aerial_eval_cli.py"

# ====== Utility Functions ======
ts() { date '+%H:%M:%S'; }

# ----- Aerial evaluation task -----
# Usage: aerial <gpu> <tag> <model_type> <checkpoint> <out_subdir> <intrinsics> <mask> <batch> [extra args...]
aerial() {
    local gpu=$1 tag=$2 mt=$3 ckpt=$4 out=$5 intr=$6 mask=$7 bsz=$8; shift 8
    local full_out="$OUTPUT_ROOT/$out"
    echo "[$(ts)] [GPU$gpu] START: $tag"
    cd "$PROJECT_ROOT"
    conda run -n "$CONDA_ENV" python "$AERIAL_CLI" \
        --model_type "$mt" --checkpoint "$ckpt" \
        --lora_config "$LORA_CONFIG" --output_dir "$full_out" \
        --gpu "$gpu" --resize 0 --batch_size "$bsz" \
        --intrinsics_mode "$intr" --mask_mode "$mask" \
        --cleanup_intermediate "$@"
    echo "[$(ts)] [GPU$gpu] DONE:  $tag"
}

# ----- Ground Base evaluation task -----
ground_base() {
    local gpu=$1 tag=$2 out_json=$3 ckpt=$4 oracle=$5
    echo "[$(ts)] [GPU$gpu] START: $tag"
    cd "$GROUND_DIR"
    local cmd="conda run -n $CONDA_ENV python moge/scripts/eval_baseline.py \
        --baseline baselines/moge2_metric.py --config $GROUND_CFG \
        --output $out_json --checkpoint $ckpt --resolution_level 9 --fp16 --device cuda:0"
    [ "$oracle" = "yes" ] && cmd="$cmd --oracle"
    eval "$cmd"
    echo "[$(ts)] [GPU$gpu] DONE:  $tag"
}

# ----- Ground LoRA evaluation task -----
ground_lora() {
    local gpu=$1 tag=$2 out_json=$3 lora_w=$4 oracle=$5
    echo "[$(ts)] [GPU$gpu] START: $tag"
    cd "$GROUND_DIR"
    local cmd="conda run -n $CONDA_ENV python moge/scripts/eval_baselinelora.py \
        --baseline baselines/moge2_lora.py --lora_config $LORA_CONFIG \
        --lora_weight $lora_w --lora_rank 96 --resolution_level 9 \
        --config $GROUND_CFG --output $out_json --device cuda:0"
    [ "$oracle" = "yes" ] && cmd="$cmd --oracle"
    eval "$cmd"
    echo "[$(ts)] [GPU$gpu] DONE:  $tag"
}

# ====== Task Queue Scheduler ======
# Each GPU slot runs one task at a time. When a task finishes, the next
# queued task immediately takes its place, maximizing GPU utilization.
# Tasks use unique output_subdir to eliminate race conditions.

schedule() {
    local -n _tasks=$1
    local total=${#_tasks[@]}
    local -a slot_pid=()
    for ((i=0; i<N_GPUS; i++)); do slot_pid[$i]=""; done

    local next=0 running=0
    while [ $next -lt $total ] || [ $running -gt 0 ]; do
        # Release finished slots
        for ((i=0; i<N_GPUS; i++)); do
            if [ -n "${slot_pid[$i]:-}" ] && ! kill -0 "${slot_pid[$i]}" 2>/dev/null; then
                slot_pid[$i]=""
                running=$((running - 1))
            fi
        done

        # Launch next task(s) on free slots
        while [ $next -lt $total ] && [ $running -lt $N_GPUS ]; do
            local slot=-1
            for ((i=0; i<N_GPUS; i++)); do
                [ -z "${slot_pid[$i]:-}" ] && { slot=$i; break; }
            done
            [ $slot -lt 0 ] && break

            local gpu="${GPU_LIST[$slot]}"
            local task="${_tasks[$next]}"

            # Parse: func|tag|arg1|arg2|...
            IFS='|' read -ra parts <<< "$task"
            local func="${parts[0]}" tag="${parts[1]}"

            (
                export CUDA_VISIBLE_DEVICES="$gpu"
                case "$func" in
                    aerial)      aerial      "$gpu" "$tag" "${parts[2]}" "${parts[3]}" "${parts[4]}" "${parts[5]}" "${parts[6]}" "${parts[7]}" "${parts[@]:8}" ;;
                    ground_base) ground_base "$gpu" "$tag" "${parts[2]}" "${parts[3]}" "${parts[4]}" ;;
                    ground_lora) ground_lora "$gpu" "$tag" "${parts[2]}" "${parts[3]}" "${parts[4]}" ;;
                esac
            ) &
            slot_pid[$slot]=$!
            next=$((next + 1))
            running=$((running + 1))
        done

        [ $next -ge $total ] && [ $running -eq 0 ] && break
        sleep 5
    done
}

# ====== Result Printing ======
print_aerial() {
    local report=$1 label=$2
    if [ -f "$report" ]; then
        echo "  === $label ==="
        grep "OVERALL" "$report" | head -1 || true
        grep "\[CAT\]" "$report" || true
        if grep -q ">>> By Scene:" "$report" 2>/dev/null; then
            echo "  --- By Scene ---"
            grep -A10 ">>> By Scene:" "$report" | grep "|" | grep -v "Group/Scene\|Group\|--" | head -6 || true
        fi
        if grep -q ">>> By Pitch:" "$report" 2>/dev/null; then
            echo "  --- By Pitch ---"
            grep -A6 ">>> By Pitch:" "$report" | grep "|" | grep -v "Group/Scene\|Group\|--" || true
        fi
        if grep -q ">>> By Height:" "$report" 2>/dev/null; then
            echo "  --- By Height ---"
            grep -A4 ">>> By Height:" "$report" | grep "|" | grep -v "Group/Scene\|--" || true
        fi
        echo ""
    else echo "  [WARN] Not found: $report"; fi
}

print_ground() {
    local f=$1 label=$2
    if [ -f "$f" ]; then
        echo "  === $label ==="
        conda run -n "$CONDA_ENV" python3 -c "
import json
with open('$f') as fh: data=json.load(fh)
for k,v in data.items():
    if isinstance(v,dict) and 'depth_metric' in v:
        m=v['depth_metric']
        print(f'  {k:12s} | AbsRel={m[\"absrel\"]:.4f}  RMSE={m[\"rmse\"]:.4f}  d1={m[\"delta1\"]:.4f}  d2={m[\"delta2\"]:.4f}')
"
        echo ""
    else echo "  [WARN] Not found: $f"; fi
}

# ====== Main ======
main() {
    echo "============================================================"
    echo " AerialMetric Benchmark — Task Queue Scheduler"
    echo " GPUs: ${GPU_LIST[*]} ($N_GPUS cards) | Start: $(date)"
    echo "============================================================"
    echo "[Pre-check]"
    conda run -n "$CONDA_ENV" python -c \
        "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    for f in "$MOGE2_BASE" "$MOGE2_AERIAL"; do [ -f "$f" ] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }; done
    for d in "$DECOUPLED" "$DECOUPLED_NORM" "$DECOUPLED_MASK" "$OBLIQUE" "$OBLIQUE_NORM" "$OBLIQUE_MASK" "$WILD"; do
        [ -d "$d" ] && echo "  OK: $d" || echo "  MISSING: $d"; done

    # Clean old results
    rm -rf "$OUTPUT_ROOT" && mkdir -p "$OUTPUT_ROOT"

    # ================================================================
    # Task queue: 10 independent tasks, each with unique output_subdir
    # Format: func|tag|arg1|arg2|...
    # ================================================================
    declare -a TASKS=(
        # --- EXP1: No Intrinsics, Mask=load, B=8 ---
        "aerial|EXP1-MoGe2-Base|full|$MOGE2_BASE|exp1_no_intrinsics|none|load|8|--decoupled_input|$DECOUPLED|--decoupled_gt|$DECOUPLED_NORM|--decoupled_csv_dir|$DECOUPLED|--decoupled_mask_dir|$DECOUPLED_MASK|--oblique_input|$OBLIQUE|--oblique_gt|$OBLIQUE|--oblique_mask_dir|$OBLIQUE_MASK|--wild_input|$WILD|--wild_gt|$WILD"
        "aerial|EXP1-Aerial-LoRA|lora96|$MOGE2_AERIAL|exp1_no_intrinsics|none|load|8|--decoupled_input|$DECOUPLED|--decoupled_gt|$DECOUPLED_NORM|--decoupled_csv_dir|$DECOUPLED|--decoupled_mask_dir|$DECOUPLED_MASK|--oblique_input|$OBLIQUE|--oblique_gt|$OBLIQUE|--oblique_mask_dir|$OBLIQUE_MASK|--wild_input|$WILD|--wild_gt|$WILD"

        # --- EXP2: With Intrinsics (norm), Mask=load ---
        #  Each sub-task uses a unique subdir to avoid race conditions
        "aerial|EXP2-Base-Decoupled(B=1)|full|$MOGE2_BASE|exp2_intrin_decoupled_base|load|load|1|--decoupled_input|$DECOUPLED_NORM|--decoupled_gt|$DECOUPLED_NORM|--decoupled_csv_dir|$DECOUPLED|--decoupled_mask_dir|$DECOUPLED_MASK"
        "aerial|EXP2-Base-Oblique(B=8)|full|$MOGE2_BASE|exp2_intrin_oblique_base|load|load|8|--oblique_input|$OBLIQUE_NORM|--oblique_gt|$OBLIQUE|--oblique_mask_dir|$OBLIQUE_MASK"
        "aerial|EXP2-Aerial-Decoupled(B=1)|lora96|$MOGE2_AERIAL|exp2_intrin_decoupled_aerial|load|load|1|--decoupled_input|$DECOUPLED_NORM|--decoupled_gt|$DECOUPLED_NORM|--decoupled_csv_dir|$DECOUPLED|--decoupled_mask_dir|$DECOUPLED_MASK"
        "aerial|EXP2-Aerial-Oblique(B=8)|lora96|$MOGE2_AERIAL|exp2_intrin_oblique_aerial|load|load|8|--oblique_input|$OBLIQUE_NORM|--oblique_gt|$OBLIQUE|--oblique_mask_dir|$OBLIQUE_MASK"

        # --- EXP3: Ground, No Oracle ---
        "ground_base|EXP3-Base-Ground|$OUTPUT_ROOT/exp3_ground/moge2_base.json|$MOGE2_BASE|no"
        "ground_lora|EXP3-Aerial-Ground|$OUTPUT_ROOT/exp3_ground/moge2_aerial.json|$MOGE2_AERIAL|no"

        # --- EXP4: Ground, With Oracle ---
        "ground_base|EXP4-Base-Ground|$OUTPUT_ROOT/exp4_ground/moge2_base_oracle.json|$MOGE2_BASE|yes"
        "ground_lora|EXP4-Aerial-Ground|$OUTPUT_ROOT/exp4_ground/moge2_aerial_oracle.json|$MOGE2_AERIAL|yes"
    )

    echo "[$(ts)] Submitting ${#TASKS[@]} tasks to $N_GPUS GPU pool..."
    schedule TASKS

    # ================================================================
    # Print all results
    # ================================================================
    echo ""; echo "=============== RESULTS SUMMARY ==============="

    B_DIR="$OUTPUT_ROOT/exp1_no_intrinsics/full/vitl-normal/Extracted"
    A_DIR="$OUTPUT_ROOT/exp1_no_intrinsics/lora96/Moge2-Aerial/Extracted"
    echo -e "\n##### EXP1: No Intrinsics, Mask=load, B=8 #####"
    echo "--- MoGe2 Base ---"
    print_aerial "$B_DIR/Decoupled/Eval_Report_Decoupled.txt" "Decoupled"
    print_aerial "$B_DIR/Oblique/Eval_Report_Oblique_Pixel.txt"   "Oblique"
    print_aerial "$B_DIR/Wild/Eval_Report_Wild_MultiRange.txt"    "Wild"
    echo "--- MoGe2 Aerial (LoRA-96) ---"
    print_aerial "$A_DIR/Decoupled/Eval_Report_Decoupled.txt" "Decoupled"
    print_aerial "$A_DIR/Oblique/Eval_Report_Oblique_Pixel.txt"   "Oblique"
    print_aerial "$A_DIR/Wild/Eval_Report_Wild_MultiRange.txt"    "Wild"

    echo -e "\n##### EXP2: With Intrinsics (norm), Mask=load #####"
    echo "--- MoGe2 Base Decoupled (B=1) ---"
    print_aerial "$OUTPUT_ROOT/exp2_intrin_decoupled_base/full/vitl-normal/Extracted/Decoupled/Eval_Report_Decoupled.txt" "Decoupled"
    echo "--- MoGe2 Base Oblique (B=8) ---"
    print_aerial "$OUTPUT_ROOT/exp2_intrin_oblique_base/full/vitl-normal/Extracted/Oblique/Eval_Report_Oblique_Pixel.txt" "Oblique"
    echo "--- MoGe2 Aerial Decoupled (B=1) ---"
    print_aerial "$OUTPUT_ROOT/exp2_intrin_decoupled_aerial/lora96/Moge2-Aerial/Extracted/Decoupled/Eval_Report_Decoupled.txt" "Decoupled"
    echo "--- MoGe2 Aerial Oblique (B=8) ---"
    print_aerial "$OUTPUT_ROOT/exp2_intrin_oblique_aerial/lora96/Moge2-Aerial/Extracted/Oblique/Eval_Report_Oblique_Pixel.txt" "Oblique"

    echo -e "\n##### EXP3: Ground, No Oracle #####"
    print_ground "$OUTPUT_ROOT/exp3_ground/moge2_base.json"   "MoGe2 Base"
    print_ground "$OUTPUT_ROOT/exp3_ground/moge2_aerial.json" "MoGe2 Aerial"

    echo -e "\n##### EXP4: Ground, With Oracle #####"
    print_ground "$OUTPUT_ROOT/exp4_ground/moge2_base_oracle.json"   "MoGe2 Base"
    print_ground "$OUTPUT_ROOT/exp4_ground/moge2_aerial_oracle.json" "MoGe2 Aerial"

    echo "All done. Results: $OUTPUT_ROOT"
}

main "$@"
