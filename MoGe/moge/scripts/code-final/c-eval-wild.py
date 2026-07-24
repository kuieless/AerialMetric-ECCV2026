
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

# Configuration

EVAL_STAGES = [400, 800] 

MIN_DEPTH = 1e-3
USE_MEDIAN_SCALING = False  # Absolute-scale evaluation.

# ===========================================

def find_gt_path_wild(pred_path, gt_root_base):
    pred_path = Path(pred_path)
    scene_name = pred_path.parent.name
    file_stem = pred_path.stem 

    gt_scene_dir = Path(gt_root_base) / scene_name
    
    possible_depth_dirs = [
        gt_scene_dir / "depth_1k",
        gt_scene_dir / "depth",
        gt_scene_dir / "depths",
        gt_scene_dir / "depth_map",
    ]
    gt_depth_dir = None
    for d in possible_depth_dirs:
        if d.exists():
            gt_depth_dir = d; break
    
    if gt_depth_dir is None: gt_depth_dir = gt_scene_dir

    candidates = [f"{file_stem}.npy", f"{file_stem}_depth.npy", f"{file_stem}.png"]
    for cand in candidates:
        target = gt_depth_dir / cand
        if target.exists(): return target
    return None

def compute_metrics(gt, pred, mask):
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    if len(gt_valid) < 10:
        return torch.full((7,), float('nan'), device=gt.device) # Seven metrics.

    # 1. AbsRel
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6))

    # 2. RMSE
    mse = ((gt_valid - pred_valid) ** 2).mean()
    rmse = torch.sqrt(mse)

    # Accuracy thresholds.
    p_safe = pred_valid.clamp(min=1e-6)
    g_safe = gt_valid.clamp(min=1e-6)
    thresh = torch.maximum((g_safe / p_safe), (p_safe / g_safe))
    
    a1 = (thresh < 1.25).float().mean()      # a1.25
    a2 = (thresh < 1.25**2).float().mean()   # a1.25 squared.
    a3 = (thresh < 1.25**3).float().mean()   # a1.25 cubed.

    # 4. N-RMSE
    g_min, g_max = gt_valid.min(), gt_valid.max()
    range_g = g_max - g_min
    if range_g < 1e-6: range_g = 1.0
    n_rmse = rmse / range_g

    # 5. SI-Log
    log_diff = torch.log(p_safe) - torch.log(g_safe)
    var = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
    si_log = torch.sqrt(torch.abs(var))

    # Order: AbsRel, RMSE, a1, a2, a3, N-RMSE, SI-Log
    return torch.tensor([abs_rel, rmse, a1, a2, a3, n_rmse, si_log])

def print_table(stats_dict, overall_list, range_cap):
    print("\n" + "="*105)
    print(f"Evaluation Range: 0 - {range_cap} meters")
    print("="*105)
    
    headers = ["Scene", "N", "AbsRel(↓)", "RMSE(↓)", "a1(↑)", "a2(↑)", "a3(↑)", "N-RMSE", "SI-Log"]
    header_fmt = "{:<22} | {:<4} | {:<9} | {:<8} | {:<7} | {:<7} | {:<7} | {:<8} | {:<6}"
    data_fmt   = "{:<22} | {:<4} | {:<9.4f} | {:<8.3f} | {:<7.3f} | {:<7.3f} | {:<7.3f} | {:<8.4f} | {:<6.3f}"
    
    print(header_fmt.format(*headers))
    print("-" * 105)

    if overall_list:
        m = np.nanmean(overall_list, axis=0)
        print(data_fmt.format("OVERALL", len(overall_list), *m))
    print("-" * 105)

    scene_avgs = []
    for s, vals in stats_dict.items():
        if vals: scene_avgs.append((s, len(vals), np.nanmean(vals, axis=0)))
    
    scene_avgs.sort(key=lambda x: x[2][0])

    for s, c, m in scene_avgs:
        s_short = (s[:20] + '..') if len(s) > 20 else s
        print(data_fmt.format(s_short, c, *m))
    
    return header_fmt, data_fmt, scene_avgs, headers

def run_wild_metric_evaluation(pred_root, gt_root_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    
    print(f"Wild Multi-Range Evaluation (Pseudo-GT Enabled)")
    print(f"   Pred: {pred_root}")
    print(f"   GT:   {gt_root_base}")
    print(f"   Ranges: {EVAL_STAGES} meters")

    pred_files = sorted(list(pred_root.rglob("*.npy")))
    multi_stage_data = {r: {'stats': defaultdict(list), 'overall': []} for r in EVAL_STAGES}

    pbar = tqdm(pred_files)
    valid_count = 0
    
    for pred_path in pbar:
        gt_path = find_gt_path_wild(pred_path, gt_root_base)
        if gt_path is None: continue

        try:
            pred_np = np.load(pred_path)
            if gt_path.suffix == '.npy': gt_np = np.load(gt_path)
            else:
                import cv2
                gt_np = cv2.imread(str(gt_path), -1)
                if gt_np is not None: gt_np = gt_np.astype(np.float32) / 1000.0
            if gt_np is None: continue
        except: continue

        pred_full = torch.from_numpy(pred_np).to(device).squeeze()
        gt_full = torch.from_numpy(gt_np).to(device).squeeze()

        if pred_full.shape != gt_full.shape:
            pred_full = F.interpolate(pred_full.unsqueeze(0).unsqueeze(0), size=gt_full.shape, mode='bilinear').squeeze()

        has_valid_data = False
        
        for max_depth in EVAL_STAGES:
            mask = (gt_full > MIN_DEPTH) & (gt_full < max_depth) & torch.isfinite(gt_full) & \
                   torch.isfinite(pred_full) & (pred_full > 0)
            
            if mask.sum() < 10: continue
            
            pred_curr = torch.clamp(pred_full, min=MIN_DEPTH, max=max_depth)

            metrics = compute_metrics(gt_full, pred_curr, mask)
            metrics_np = metrics.cpu().numpy()

            if np.isnan(metrics_np).any(): continue

            scene_name = pred_path.parent.name
            multi_stage_data[max_depth]['stats'][scene_name].append(metrics_np)
            multi_stage_data[max_depth]['overall'].append(metrics_np)
            has_valid_data = True
            
        if has_valid_data:
            valid_count += 1
            pbar.set_postfix({"Valid": valid_count})

    out_file = pred_root / "Eval_Report_Wild_MultiRange.txt"
    f_handle = open(out_file, "w")
    f_handle.write(f"Wild Multi-Range Evaluation (Pseudo-GT)\nInput: {pred_root}\nGT: {gt_root_base}\n")

    for max_depth in EVAL_STAGES:
        data = multi_stage_data[max_depth]
        header_fmt, data_fmt, scene_avgs, headers = print_table(data['stats'], data['overall'], max_depth)
        
        f_handle.write(f"\n\n=== Range: 0 - {max_depth}m ===\n")
        f_handle.write(header_fmt.format(*headers) + "\n")
        f_handle.write("-" * 105 + "\n")
        if data['overall']:
            m = np.nanmean(data['overall'], axis=0)
            f_handle.write(data_fmt.format("OVERALL", len(data['overall']), *m) + "\n")
        f_handle.write("-" * 105 + "\n")
        for s, c, m in scene_avgs:
            f_handle.write(data_fmt.format(s, c, *m) + "\n")

    f_handle.close()
    print(f"\nFull Report saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()
    
    if os.path.exists(args.pred):
        run_wild_metric_evaluation(args.pred, args.gt)
    else:
        print("Pred path not found")
