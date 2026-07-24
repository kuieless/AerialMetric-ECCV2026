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

# Map dataset names to CSV metadata filenames
# Keys must match scene folder names.
# CSV metadata files are expected in GT_ROOT/<scene>/final_dataset_<scene_lower>.csv
# Override via --csv_dir if your CSVs are stored elsewhere.
CSV_MAP_TEMPLATE = {
    "Cleaned_Dataset_Campus":  "final_dataset_campus.csv",
    "Cleaned_Dataset_Factory": "final_dataset_factory.csv",
    "Cleaned_Dataset_Farm":    "final_dataset_farm.csv",
    "Cleaned_Dataset_Gress":   "final_dataset_grass.csv",
}

# Evaluation settings, aligned with Oblique evaluation.
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False

# ===========================================

def load_csv_metadata(csv_path):
    """Read CSV metadata and build Filename_Stem -> {pitch, height} index."""
    try:
        # Try UTF-8 first, then GBK for legacy CSV files.
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        meta_dict = {}
        # Validate required columns with fallbacks.
        target_col = '\u5339\u914d\u5230\u7684\u53c2\u8003\u56fe(Target)'
        pitch_col = '\u5b9e\u9645Pitch'
        height_col = '\u76f8\u5bf9\u9ad8\u5ea6(\u81ea\u52a8)'

        if target_col not in df.columns:
            print(f"Warning: CSV {csv_path} missing '{target_col}' column")
            return {}

        for _, row in df.iterrows():
            target_name = str(row[target_col]) # e.g., "abcde.JPG"
            stem = os.path.splitext(target_name)[0] # "abcde"
            
            pitch = row[pitch_col] if pitch_col in df.columns else -90
            height = row[height_col] if height_col in df.columns else 0
            
            meta_dict[stem] = {
                'pitch': float(pitch),
                'height': float(height)
            }
        return meta_dict

    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return {}

def find_gt_path_bench(pred_path, gt_root_base):
    """Find the matching Decoupled GT path."""
    scene_dir_name = pred_path.parent.name # Cleaned_Dataset_Campus
    file_name = pred_path.name # ImageHash.npy
    stem = pred_path.stem

    # Decoupled norm-style format: GT_ROOT/Scene/SampleID/depth.npy
    gt_path_sample = Path(gt_root_base) / scene_dir_name / stem / "depth.npy"
    if gt_path_sample.exists():
        return gt_path_sample
    
    # Legacy flattened format: GT_ROOT/Scene/depth/SampleID.npy
    gt_path = Path(gt_root_base) / scene_dir_name / "depth" / file_name
    if gt_path.exists():
        return gt_path
    
    # Fallback for *_depth.npy naming.
    gt_path_alt = Path(gt_root_base) / scene_dir_name / "depth" / f"{stem}_depth.npy"
    if gt_path_alt.exists():
        return gt_path_alt
        
    return None

def find_mask_path_bench(pred_path, mask_root_base):
    scene_dir_name = pred_path.parent.name
    stem = pred_path.stem
    mask_root = Path(mask_root_base)
    candidates = [
        mask_root / f"{scene_dir_name}-mask" / f"{stem}.png",
        mask_root / scene_dir_name / f"{stem}.png",
        mask_root / f"{scene_dir_name}.png",
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
    """Least-squares scale-shift alignment."""
    safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    t_valid = target[safe_mask]
    p_valid = pred[safe_mask]
    if len(t_valid) < 10: return pred
    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
    except: s, t = 1.0, 0.0
    if s <= 1e-4 or np.isnan(s): s = 1.0
    return pred * s + t

def compute_metrics(gt, pred, mask):
    """Compute metrics, aligned with Oblique evaluation."""
    gt = gt[mask]
    pred = pred[mask]
    
    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1_10 = (thresh < 1.10).float().mean()
    a1_25 = (thresh < 1.25).float().mean()
    
    # SI-Log
    log_diff = torch.log(pred) - torch.log(gt)
    si_log = torch.sqrt(torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2)

    # N-RMSE
    p_min, p_max = pred.min(), pred.max()
    g_min, g_max = gt.min(), gt.max()
    p_norm = (pred - p_min) / (p_max - p_min + 1e-8)
    g_norm = (gt - g_min) / (g_max - g_min + 1e-8)
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

    # Spearman
    p_cpu = pred.detach().cpu().numpy()
    g_cpu = gt.detach().cpu().numpy()
    if len(p_cpu) > 5000:
        idx = np.random.choice(len(p_cpu), 5000, replace=False)
        p_cpu = p_cpu[idx]
        g_cpu = g_cpu[idx]
    
    if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
        spearman = 0.0
    else:
        spearman, _ = spearmanr(p_cpu, g_cpu)

    # 0:AbsRel, 1:RMSE, 2:a1.10, 3:a1.25, 4:SI-Log, 5:Spearman, 6:N-RMSE
    return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse])

def run_bench_evaluation(pred_root, gt_root_base, csv_dir=None, mask_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    csv_dir = Path(csv_dir) if csv_dir else Path(gt_root_base)
    
    print(f"Starting Decoupled evaluation")
    print(f"Pred Root: {pred_root}")
    print(f"GT Root:   {gt_root_base}")
    print(f"CSV Dir:   {csv_dir}")
    if mask_dir:
        print(f"Mask Dir:  {mask_dir}")

    # Preload CSV metadata.
    full_meta = {}
    print("\n📖 Loading CSV metadata...")
    for folder_name, csv_filename in CSV_MAP_TEMPLATE.items():
        csv_path = csv_dir / csv_filename
        if os.path.exists(csv_path):
            full_meta[folder_name] = load_csv_metadata(csv_path)
            print(f"  - {folder_name}: loaded {len(full_meta[folder_name])} records")
        else:
            print(f"  WARNING: CSV not found: {csv_path}")
            full_meta[folder_name] = {}

    # Scan prediction files.
    pred_files = sorted(list(pred_root.rglob("*.npy")))
    print(f"\nFound {len(pred_files)} prediction files. Starting matching...")

    # Metric accumulators.
    overall_stats = []
    scene_stats = defaultdict(list)
    pitch_stats = defaultdict(list)  # Group by pitch angle.
    height_stats = defaultdict(list) # Group by altitude.
    
    # Per-image detailed records.
    per_image_records = [] 

    pbar = tqdm(pred_files)
    valid_count = 0
    
    for pred_path in pbar:
        scene_name = pred_path.parent.name # Cleaned_Dataset_Campus
        stem = pred_path.stem

        # Step A: find GT.
        gt_path = find_gt_path_bench(pred_path, gt_root_base)
        if gt_path is None: continue
        mask_path = find_mask_path_bench(pred_path, mask_dir) if mask_dir else None

        # Step B: find CSV metadata.
        meta = full_meta.get(scene_name, {}).get(stem, None)
        pitch = meta['pitch'] if meta else -90.0
        height = meta['height'] if meta else 0.0

        # Step C: load arrays and compute metrics.
        try:
            pred_np = np.load(pred_path)
            gt_np = np.load(gt_path)
        except: continue
        
        pred = torch.from_numpy(pred_np).to(device).squeeze()
        gt = torch.from_numpy(gt_np).to(device).squeeze()
        
        if pred.shape != gt.shape:
             pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()

        external_invalid_mask = load_invalid_mask(mask_path, gt.shape, device) if mask_path is not None else None
        
        mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt) & torch.isfinite(pred) & (pred > 0)
        if external_invalid_mask is not None:
            mask = mask & (~external_invalid_mask)
        if mask.sum() < 10: continue

        if USE_MEDIAN_SCALING:
            pred = align_scale_shift_torch(pred, gt, mask)
        pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

        metrics = compute_metrics(gt, pred, mask) # Tensor
        metrics_np = metrics.cpu().numpy()

        # Step D: aggregate statistics.
        overall_stats.append(metrics_np)
        scene_stats[scene_name].append(metrics_np)
        
        # Pitch grouping.
        if pitch <= -85: p_key = "Nadir (-90)"
        elif pitch <= -60: p_key = "Oblique (-60 to -85)"
        elif pitch <= -45: p_key = "High Oblique (-45 to -60)"
        else: p_key = "Horizontal (>-45)"
        pitch_stats[p_key].append(metrics_np)

        # Height grouping.
        if height < 60: h_key = "Low (<60m)"
        elif height < 120: h_key = "Mid (60-120m)"
        else: h_key = "High (>120m)"
        height_stats[h_key].append(metrics_np)
        
        # Add per-image detail record.
        per_image_records.append({
            "Scene": scene_name,
            "Filename": stem,
            "Pitch": pitch,
            "Pitch_Group": p_key,
            "Height": height,
            "Height_Group": h_key,
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

    # Generate summary report.
    
    headers = ["Group/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
    header_fmt = "{:<28} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
    data_fmt   = "{:<28} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
    print("\n" + "="*125)
    print(header_fmt.format(*headers))
    print("-" * 125)

    if overall_stats:
        m = np.mean(overall_stats, axis=0)
        print(data_fmt.format("OVERALL", len(overall_stats), *m))
    print("-" * 125)

    print(">>> By Scene Dataset:")
    for s in sorted(scene_stats.keys()):
        m = np.mean(scene_stats[s], axis=0)
        print(data_fmt.format(s, len(scene_stats[s]), *m))
    print("-" * 125)

    print(">>> By Pitch (Angle):")
    pitch_order = ["Nadir (-90)", "Oblique (-60 to -85)", "High Oblique (-45 to -60)", "Horizontal (>-45)"]
    for p in pitch_order:
        if p in pitch_stats:
            m = np.mean(pitch_stats[p], axis=0)
            print(data_fmt.format(p, len(pitch_stats[p]), *m))
    print("-" * 125)

    print(">>> By Height (Altitude):")
    height_order = ["Low (<60m)", "Mid (60-120m)", "High (>120m)"]
    for h in height_order:
        if h in height_stats:
            m = np.mean(height_stats[h], axis=0)
            print(data_fmt.format(h, len(height_stats[h]), *m))

    # Save summary report.
    output_file = pred_root / "Eval_Report_Decoupled.txt"
    with open(output_file, "w") as f:
        f.write(f"Decoupled Evaluation\nPred: {pred_root}\nGT: {gt_root_base}\nCSV: {csv_dir}\n\n")
        f.write(header_fmt.format(*headers) + "\n")
        f.write("-" * 125 + "\n")
        
        if overall_stats:
            m = np.mean(overall_stats, axis=0)
            f.write(data_fmt.format("OVERALL", len(overall_stats), *m) + "\n")
        f.write("-" * 125 + "\n")
        
        for name, stats_dict in [("By Scene", scene_stats), ("By Pitch", pitch_stats), ("By Height", height_stats)]:
            f.write(f">>> {name}:\n")
            for k in sorted(stats_dict.keys()):
                m = np.mean(stats_dict[k], axis=0)
                f.write(data_fmt.format(k, len(stats_dict[k]), *m) + "\n")
            f.write("-" * 125 + "\n")

    print(f"\nSummary report saved: {output_file}")

    # Export per-image details to CSV.
    if per_image_records:
        df_detailed = pd.DataFrame(per_image_records)
        detailed_output_file = pred_root / "Eval_Report_Decoupled_Detailed.csv"
        # Save CSV with four decimal places.
        df_detailed.to_csv(detailed_output_file, index=False, float_format="%.4f", encoding='utf-8-sig')
        print(f"Per-image details saved: {detailed_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CLI arguments.
    parser.add_argument("--pred", required=True, help="Path to extracted predictions (flattened .npy files)")
    parser.add_argument("--gt", required=True, help="Path to Decoupled GT dataset root")
    parser.add_argument("--csv_dir", default=None, help="Path to Decoupled CSV metadata. Defaults to --gt.")
    parser.add_argument("--mask_dir", default=None, help="Path to Decoupled mask root with white invalid pixels")
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"ERROR: prediction directory does not exist {args.pred}")
    else:
        run_bench_evaluation(args.pred, args.gt, args.csv_dir, args.mask_dir)
