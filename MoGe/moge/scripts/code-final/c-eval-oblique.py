import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

# Configuration

SCENE_CATEGORIES = {
    "Rural": [
        "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output",
        "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled",
        "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled",
        "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled",
        "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output",
        "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled", "BC2", "BC1", "L1"
    ],
    "Natural": [
        "lewis-output", "park5", "park13", "park14", "park10", "park0",
        "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled",
        "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled",
        "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled",
        "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled",
        "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled",
        "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"
    ],
    "City": [
        "yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"
    ],
    "Factory": [
        "factory_scene_1", "factory_scene_2"
    ]
}

# Evaluation settings
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False

# ===========================================

def get_category(scene_name):
    for cat, scenes in SCENE_CATEGORIES.items():
        if any(s in scene_name for s in scenes):
            return cat
    return "Uncategorized"

def find_gt_path(pred_path, gt_root_base):
    pred_path = Path(pred_path)
    scene_name = pred_path.parent.name
    file_stem = pred_path.stem 

    gt_scene_dir = Path(gt_root_base) / scene_name
    possible_depth_dirs = [gt_scene_dir / "depth", gt_scene_dir / "depths"]
    
    gt_depth_dir = None
    for d in possible_depth_dirs:
        if d.exists():
            gt_depth_dir = d
            break
    
    if gt_depth_dir is None:
        return None

    candidates = [
        f"{file_stem}.npy",
        f"{file_stem}_depth.npy",
        f"{file_stem}.png",
        f"{file_stem}_depth.png"
    ]

    for cand in candidates:
        target = gt_depth_dir / cand
        if target.exists():
            return target
            
    return None

def find_mask_path_oblique(pred_path, mask_root_base):
    scene_name = pred_path.parent.name
    stem = pred_path.stem
    mask_root = Path(mask_root_base)
    candidates = [
        mask_root / f"{scene_name}-mask" / f"{stem}.png",
        mask_root / scene_name / f"{stem}.png",
        mask_root / f"{scene_name}.png",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None

def load_invalid_mask(mask_path, target_shape, device):
    if mask_path is None:
        return None
    mask_np = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_np is None:
        return None
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    invalid_np = mask_np > 127
    mask = torch.from_numpy(invalid_np).to(device)
    if mask.shape != target_shape:
        mask = F.interpolate(
            mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode='nearest',
        ).squeeze(0).squeeze(0) > 0.5
    return mask.bool()

def align_scale_shift_torch(pred, target, mask):
    # Estimate scale and shift from valid pixels only.
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
        
    # Fall back to identity alignment for invalid least-squares results.
    if np.isnan(s) or np.isinf(s) or s <= 1e-6:
        s, t = 1.0, 0.0
        
    return pred * s + t

def compute_metrics(gt, pred, mask):
    """
    Compute metrics using only pixels selected by the mask.
    """
    # Flatten valid pixels.
    # Invalid pixels are discarded here.
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # Not enough valid pixels for stable metrics.
    if len(gt_valid) < 10:
        return torch.full((9,), float('nan'), device=gt.device)

    # Metrics are computed on 1D valid-pixel vectors.
    
    # Basic errors.
    mse = ((gt_valid - pred_valid) ** 2).mean()
    rmse = torch.sqrt(mse)
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6))
    
    # Threshold accuracies.
    # Clamp defensively to avoid division by zero.
    p_safe = pred_valid.clamp(min=1e-6)
    g_safe = gt_valid.clamp(min=1e-6)
    thresh = torch.maximum((g_safe / p_safe), (p_safe / g_safe))
    
    a1_10 = (thresh < 1.10).float().mean()
    a1_25 = (thresh < 1.25).float().mean()
    a2    = (thresh < 1.25 ** 2).float().mean()
    a3    = (thresh < 1.25 ** 3).float().mean()

    # (c) SI-Log
    # Non-positive values have already been masked out.
    log_pred = torch.log(p_safe)
    log_gt = torch.log(g_safe)
    log_diff = log_pred - log_gt
    
    # Avoid negative variance from floating-point roundoff.
    var = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
    si_log = torch.sqrt(torch.abs(var))

    # (d) N-RMSE
    # Normalize with guards for constant predictions or targets.
    p_min, p_max = pred_valid.min(), pred_valid.max()
    g_min, g_max = gt_valid.min(), gt_valid.max()
    
    range_p = p_max - p_min
    range_g = g_max - g_min
    
    if range_p < 1e-6: range_p = 1.0
    if range_g < 1e-6: range_g = 1.0
    
    p_norm = (pred_valid - p_min) / range_p
    g_norm = (gt_valid - g_min) / range_g
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

    # (e) Spearman
    p_cpu = pred_valid.detach().cpu().numpy()
    g_cpu = gt_valid.detach().cpu().numpy()
    
    # Subsample to keep Spearman computation cheap.
    if len(p_cpu) > 5000:
        idx = np.random.choice(len(p_cpu), 5000, replace=False)
        p_cpu = p_cpu[idx]
        g_cpu = g_cpu[idx]
    
    # Spearman is undefined for constant arrays.
    if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
        spearman = 0.0
    else:
        spearman, _ = spearmanr(p_cpu, g_cpu)
        if np.isnan(spearman): spearman = 0.0

    return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3])

def run_evaluation(pred_root, gt_root_base, mask_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    
    print(f"Scanning predictions: {pred_root}")
    if mask_dir:
        print(f"Mask Root: {mask_dir}")
    pred_files = sorted(list(pred_root.rglob("*.npy")))
    print(f"Found {len(pred_files)} prediction files. Matching GT...")

    stats = defaultdict(list)
    category_stats = defaultdict(list)
    overall_stats = []
    
    # Per-image detailed records.
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
        mask_path = find_mask_path_oblique(pred_path, mask_dir) if mask_dir else None

        try:
            pred_np = np.load(pred_path)
            # NaN values are handled by the metric mask below.
            
            if gt_path.suffix == '.npy':
                gt_np = np.load(gt_path)
            else:
                import cv2
                gt_np = cv2.imread(str(gt_path), -1)
                if gt_np is not None: gt_np = gt_np.astype(np.float32) / 1000.0
            
            if gt_np is None: continue

        except Exception as e:
            print(f"Error reading {pred_path.name}: {e}")
            continue

        pred = torch.from_numpy(pred_np).to(device).squeeze()
        gt = torch.from_numpy(gt_np).to(device).squeeze()

        if pred.shape != gt.shape:
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()

        external_invalid_mask = load_invalid_mask(mask_path, gt.shape, device) if mask_path is not None else None

        # Build a strict valid-pixel mask.
        # GT must be finite and in range.
        mask_gt = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
        # Prediction must be finite and positive.
        mask_pred = torch.isfinite(pred) & (pred > 0)
        # Combine masks.
        mask = mask_gt & mask_pred
        if external_invalid_mask is not None:
            mask = mask & (~external_invalid_mask)
        
        # Require at least 10 valid pixels.
        if mask.sum() < 10: 
            # Skip images with too few valid pixels.
            continue

        # Optional scale-shift alignment using valid pixels only.
        if USE_MEDIAN_SCALING:
            pred = align_scale_shift_torch(pred, gt, mask)
            # Clamp after alignment.
            pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

        # Compute metrics.
        metrics = compute_metrics(gt, pred, mask)
        metrics_np = metrics.cpu().numpy()

        # Skip rare invalid metric outputs.
        if np.isnan(metrics_np).any():
            # print(f"NaN Metrics: {pred_path.name}")
            continue

        scene_name = pred_path.parent.name
        cat_name = get_category(scene_name)
        file_stem = pred_path.stem
        
        if cat_name == "Uncategorized":
            uncategorized_scenes.add(scene_name)

        stats[scene_name].append(metrics_np)
        category_stats[cat_name].append(metrics_np)
        overall_stats.append(metrics_np)
        
        # Add per-image detail record.
        # Metric order: [abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3].
        per_image_records.append({
            "Category": cat_name,
            "Scene": scene_name,
            "Filename": file_stem,
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

    # Generate reports.
    
    print("\n" + "="*120)
    print(f"Evaluation complete. Valid samples: {valid_count}, missing GT: {missing_gt_count}")
    print("="*120)

    if uncategorized_scenes:
        print(f" Uncategorized scenes: {uncategorized_scenes}")
        print("-" * 120)
    
    headers = ["Category/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
    header_fmt = "{:<22} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
    data_fmt   = "{:<22} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
    print(header_fmt.format(*headers))
    print("-" * 120)

    # Use nanmean to avoid rare NaNs contaminating aggregate metrics.
    
    # 1. Overall
    if overall_stats:
        m = np.nanmean(overall_stats, axis=0)
        print(data_fmt.format("OVERALL", len(overall_stats), *m[:7]))
    print("-" * 120)

    # 2. Category
    for cat in sorted(category_stats.keys()):
        if len(category_stats[cat]) > 0:
            m = np.nanmean(category_stats[cat], axis=0)
            print(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]))
    print("-" * 120)

    # Scene-level metrics sorted by AbsRel.
    scene_avgs = []
    for s, vals in stats.items():
        if len(vals) == 0: continue
        scene_avgs.append((s, len(vals), np.nanmean(vals, axis=0)))
    
    # Sort by AbsRel.
    scene_avgs.sort(key=lambda x: x[2][0]) 

    print(">>> Top 10 Scenes (Sorted by AbsRel):")
    for s, c, m in scene_avgs[:10]:
         s_short = (s[:20] + '..') if len(s) > 20 else s
         print(data_fmt.format(s_short, c, *m[:7]))

    # Save text report.
    output_file = pred_root / "Eval_Report_Oblique_Pixel.txt"
    with open(output_file, "w") as f:
        f.write(f"Evaluation Date: {np.datetime64('now')}\n")
        
        if uncategorized_scenes:
            f.write(f"\nUncategorized Scenes: {uncategorized_scenes}\n\n")

        f.write(header_fmt.format(*headers) + "\n")
        f.write("-" * 120 + "\n")
        if overall_stats:
            m = np.nanmean(overall_stats, axis=0)
            f.write(data_fmt.format("OVERALL", len(overall_stats), *m[:7]) + "\n")
        f.write("-" * 120 + "\n")
        for cat in sorted(category_stats.keys()):
            if len(category_stats[cat]) > 0:
                m = np.nanmean(category_stats[cat], axis=0)
                f.write(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]) + "\n")
        f.write("-" * 120 + "\n")
        f.write(">>> All Scenes (Sorted by AbsRel):\n")
        for s, c, m in scene_avgs:
            f.write(data_fmt.format(s, c, *m[:7]) + "\n")
            
    print(f"\nFull report saved to: {output_file}")
    
    # Export per-image details to CSV.
    if per_image_records:
        df_detailed = pd.DataFrame(per_image_records)
        detailed_output_file = pred_root / "Eval_Report_Oblique_Pixel_Detailed.csv"
        # Save CSV with four decimal places.
        df_detailed.to_csv(detailed_output_file, index=False, float_format="%.4f", encoding='utf-8-sig')
        print(f"Per-image details saved to: {detailed_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to extracted predictions directory")
    parser.add_argument("--gt", required=True, help="Path to Oblique GT dataset root")
    parser.add_argument("--mask_dir", default=None, help="Path to Oblique mask root with white invalid pixels")
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"ERROR: directory does not exist {args.pred}")
    else:
        run_evaluation(args.pred, args.gt, args.mask_dir)
