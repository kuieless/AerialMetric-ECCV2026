import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

# ================= 配置区域 =================

# 更新了分类映射，匹配你的 Cleaned_Dataset 系列
SCENE_CATEGORIES = {
    "Rural": ["Farm", "Rural"],
    "Natural": ["Gress", "Natural"],
    "City": ["Campus", "City"],
    "Factory": ["Factory"]
}

# 评估参数
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False # 如果是 Metric Depth 模型（如你之前的 AeroMetric 实验），通常设为 False

# ===========================================

def get_category(scene_name):
    for cat, keywords in SCENE_CATEGORIES.items():
        if any(kw in scene_name for kw in keywords):
            return cat
    return "Uncategorized"

def find_gt_path(pred_path, gt_root_base):
    """
    匹配逻辑更新：
    pred: .../Extracted/Cleaned_Dataset_Campus/file.npy
    gt:   .../Val/Bench/Cleaned_Dataset_Campus/depth/file.npy
    """
    pred_path = Path(pred_path)
    scene_name = pred_path.parent.name  # 例如: Cleaned_Dataset_Campus
    file_stem = pred_path.stem         # 文件名（不含后缀）

    # gt_root_base should point to the Bench GT dataset root
    gt_scene_dir = Path(gt_root_base) / scene_name
    
    # 根据你提供的结构，GT 都在 depth 文件夹下
    gt_depth_dir = gt_scene_dir / "depth"
    
    if not gt_depth_dir.exists():
        # 兼容性检查：尝试一下 depths (复数)
        if (gt_scene_dir / "depths").exists():
            gt_depth_dir = gt_scene_dir / "depths"
        else:
            return None

    # 尝试多种可能的后缀
    candidates = [
        f"{file_stem}.npy",
        f"{file_stem}.png",
        f"{file_stem}_depth.npy",
        f"{file_stem}_depth.png"
    ]

    for cand in candidates:
        target = gt_depth_dir / cand
        if target.exists():
            return target
            
    return None

def align_scale_shift_torch(pred, target, mask):
    t_valid = target[mask]
    p_valid = pred[mask]
    if len(t_valid) < 10: return pred
    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
    except:
        s, t = 1.0, 0.0
    if np.isnan(s) or np.isinf(s) or s <= 1e-6:
        s, t = 1.0, 0.0
    return pred * s + t

def compute_metrics(gt, pred, mask):
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    if len(gt_valid) < 10:
        return torch.full((9,), float('nan'), device=gt.device)

    # 基础误差
    mse = ((gt_valid - pred_valid) ** 2).mean()
    rmse = torch.sqrt(mse)
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6))
    
    # 准确率
    p_safe = pred_valid.clamp(min=1e-6)
    g_safe = gt_valid.clamp(min=1e-6)
    thresh = torch.maximum((g_safe / p_safe), (p_safe / g_safe))
    
    a1_10 = (thresh < 1.10).float().mean()
    a1_25 = (thresh < 1.25).float().mean()
    a2    = (thresh < 1.25 ** 2).float().mean()
    a3    = (thresh < 1.25 ** 3).float().mean()

    # SI-Log
    log_pred = torch.log(p_safe)
    log_gt = torch.log(g_safe)
    log_diff = log_pred - log_gt
    var = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
    si_log = torch.sqrt(torch.abs(var))

    # N-RMSE
    p_min, p_max = pred_valid.min(), pred_valid.max()
    g_min, g_max = gt_valid.min(), gt_valid.max()
    range_p = (p_max - p_min).clamp(min=1e-6)
    range_g = (g_max - g_min).clamp(min=1e-6)
    p_norm = (pred_valid - p_min) / range_p
    g_norm = (gt_valid - g_min) / range_g
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

    # Spearman
    p_cpu = pred_valid.detach().cpu().numpy()
    g_cpu = gt_valid.detach().cpu().numpy()
    if len(p_cpu) > 5000:
        idx = np.random.choice(len(p_cpu), 5000, replace=False)
        p_cpu, g_cpu = p_cpu[idx], g_cpu[idx]
    
    if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
        spearman = 0.0
    else:
        spearman, _ = spearmanr(p_cpu, g_cpu)
    
    return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3])

def run_evaluation(pred_root, gt_root_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    
    print(f"🕵️  正在扫描预测结果: {pred_root}")
    # 递归查找所有 npy，但排除掉已经生成的 CSV/TXT 报告文件
    pred_files = sorted([f for f in pred_root.rglob("*.npy") if "Eval_Report" not in f.name])
    print(f"✅ 找到 {len(pred_files)} 个预测文件，开始匹配 GT...")

    stats = defaultdict(list)
    category_stats = defaultdict(list)
    overall_stats = []
    per_image_records = []
    uncategorized_scenes = set()
    
    pbar = tqdm(pred_files)
    valid_count = 0
    missing_gt_count = 0

    for pred_path in pbar:
        gt_path = find_gt_path(pred_path, gt_root_base)
        if gt_path is None:
            missing_gt_count += 1
            continue

        try:
            pred_np = np.load(pred_path)
            if gt_path.suffix == '.npy':
                gt_np = np.load(gt_path)
            else:
                import cv2
                gt_np = cv2.imread(str(gt_path), -1)
                if gt_np is not None: gt_np = gt_np.astype(np.float32) / 1000.0
            
            if gt_np is None: continue

            pred = torch.from_numpy(pred_np).to(device).squeeze()
            gt = torch.from_numpy(gt_np).to(device).squeeze()

            if pred.shape != gt.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()

            mask_gt = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
            mask_pred = torch.isfinite(pred) & (pred > 0)
            mask = mask_gt & mask_pred
            
            if mask.sum() < 10: continue

            if USE_MEDIAN_SCALING:
                pred = align_scale_shift_torch(pred, gt, mask)
                pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

            metrics = compute_metrics(gt, pred, mask)
            metrics_np = metrics.cpu().numpy()

            if np.isnan(metrics_np).any(): continue

            scene_name = pred_path.parent.name
            cat_name = get_category(scene_name)
            
            if cat_name == "Uncategorized":
                uncategorized_scenes.add(scene_name)

            stats[scene_name].append(metrics_np)
            category_stats[cat_name].append(metrics_np)
            overall_stats.append(metrics_np)
            
            per_image_records.append({
                "Category": cat_name,
                "Scene": scene_name,
                "Filename": pred_path.stem,
                "AbsRel": metrics_np[0],
                "RMSE": metrics_np[1],
                "a1.10": metrics_np[2],
                "a1.25": metrics_np[3],
                "SI-Log": metrics_np[4],
                "Spearman": metrics_np[5],
                "N-RMSE": metrics_np[6]
            })
            
            valid_count += 1
            pbar.set_postfix({"Valid": valid_count})

        except Exception as e:
            print(f"Error processing {pred_path.name}: {e}")
            continue

    # ================= 生成报告 =================
    print("\n" + "="*120)
    print(f"📊 评估完成! 有效样本: {valid_count}, 缺失GT: {missing_gt_count}")
    
    headers = ["Category/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
    header_fmt = "{:<22} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
    data_fmt   = "{:<22} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
    print("-" * 120)
    print(header_fmt.format(*headers))
    print("-" * 120)

    if overall_stats:
        m = np.nanmean(overall_stats, axis=0)
        print(data_fmt.format("OVERALL", len(overall_stats), *m[:7]))

    for cat in sorted(category_stats.keys()):
        if len(category_stats[cat]) > 0:
            m = np.nanmean(category_stats[cat], axis=0)
            print(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]))

    output_file = pred_root / "Eval_Report_Bench.txt"
    detailed_output_file = pred_root / "Eval_Report_Bench_Detailed.csv"

    # 保存 TXT 报告
    with open(output_file, "w") as f:
        f.write(header_fmt.format(*headers) + "\n")
        if overall_stats:
            m = np.nanmean(overall_stats, axis=0)
            f.write(data_fmt.format("OVERALL", len(overall_stats), *m[:7]) + "\n")
        for cat in sorted(category_stats.keys()):
            if len(category_stats[cat]) > 0:
                m = np.nanmean(category_stats[cat], axis=0)
                f.write(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]) + "\n")

    # 保存 CSV 详细数据
    if per_image_records:
        pd.DataFrame(per_image_records).to_csv(detailed_output_file, index=False, float_format="%.4f", encoding='utf-8-sig')

    print(f"\n📝 报告已保存至:\n1. {output_file}\n2. {detailed_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值改为你现在的 Extracted 目录
    parser.add_argument("--pred", required=True, help="Path to extracted predictions directory")
    parser.add_argument("--gt", required=True, help="Path to Bench GT dataset root")
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"❌ 错误: 目录不存在 {args.pred}")
    else:
        run_evaluation(args.pred, args.gt)