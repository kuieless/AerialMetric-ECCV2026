import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # Required for interpolation.
import math
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

# Configuration
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False 
# =======================================

def calculate_fov_from_focal(fx, width):
    fov_rad = 2 * np.arctan(width / (2 * fx))
    return np.degrees(fov_rad)

def load_scene_metadata(scene_dir):
    csv_path = scene_dir / "metadata_full.csv"
    if not csv_path.exists():
        # Debug: print missing CSV paths.
        # print(f"   [Debug] CSV not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        meta_dict = {}
        for _, row in df.iterrows():
            fname = row.get('filename_npy', row.get('filename_img', ''))
            stem = Path(fname).stem
            if 'fx' in row and 'width' in row:
                meta_dict[stem] = {
                    'fx': float(row['fx']),
                    'width': int(row['width'])
                }
        return meta_dict
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None

def compute_depth_metrics(gt, pred, mask):
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    if len(gt_valid) < 10: return None
    rmse = torch.sqrt(((gt_valid - pred_valid) ** 2).mean()).item()
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6)).item()
    return {'rmse': rmse, 'abs_rel': abs_rel}

def run_fov_evaluation(pred_root_base, gt_root_base):
    pred_root_base = Path(pred_root_base)
    gt_root_base = Path(gt_root_base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"FoV & Depth Joint Evaluation")
    print(f"   Pred Root: {pred_root_base}")
    print(f"   GT Root:   {gt_root_base}")

    fov_files = sorted(list(pred_root_base.rglob("fov.json")))
    print(f"Found {len(fov_files)} fov.json files.")

    scene_cache = {}
    results = []
    
    # Failure reason counters.
    fail_reasons = defaultdict(int)

    pbar = tqdm(fov_files)
    for fov_path in pbar:
        # Path parsing logic.
        image_dir = fov_path.parent
        image_stem = image_dir.name 
        scene_dir_name = image_dir.parent.name 
        
        # Debug 1: check scene folder.
        gt_scene_dir = gt_root_base / scene_dir_name
        if not gt_scene_dir.exists(): 
            fail_reasons[f"Missing Scene Dir: {scene_dir_name}"] += 1
            continue 

        # Debug 2: load metadata.
        if scene_dir_name not in scene_cache:
            scene_cache[scene_dir_name] = load_scene_metadata(gt_scene_dir)
        
        meta_dict = scene_cache[scene_dir_name]
        if meta_dict is None:
            fail_reasons[f"No CSV in {scene_dir_name}"] += 1
            continue
            
        if image_stem not in meta_dict:
            fail_reasons[f"Image not in CSV: {image_stem}"] += 1
            continue 

        # Debug 3: load GT depth.
        gt_path = gt_scene_dir / "depth_1k" / f"{image_stem}.npy"
        if not gt_path.exists():
            gt_path = gt_scene_dir / "depth" / f"{image_stem}.npy"
        if not gt_path.exists(): 
            gt_path = gt_scene_dir / "depth" / f"{image_stem}_depth.npy"
            if not gt_path.exists(): 
                fail_reasons[f"Missing GT NPY"] += 1
                continue

        # Step B: load data.
        try:
            with open(fov_path, 'r') as f:
                pred_fov_data = json.load(f)
                pred_fov_x = pred_fov_data['fov_x']

            gt_fx = meta_dict[image_stem]['fx']
            gt_w = meta_dict[image_stem]['width']
            gt_fov_x = calculate_fov_from_focal(gt_fx, gt_w)

            fov_diff = pred_fov_x - gt_fov_x
            fov_err_abs = abs(fov_diff)

            pred_depth_path = image_dir / "depth.npy"
            pred_np = np.load(pred_depth_path)
            gt_np = np.load(gt_path)
        except Exception as e: 
            fail_reasons[f"Load Error: {e}"] += 1
            continue

        # Step C: compute metrics.
        pred_t = torch.from_numpy(pred_np).to(device).squeeze()
        gt_t = torch.from_numpy(gt_np).to(device).squeeze()

        # F.interpolate is available through torch.nn.functional.
        if pred_t.shape != gt_t.shape:
            pred_t = F.interpolate(pred_t.unsqueeze(0).unsqueeze(0), size=gt_t.shape, mode='bilinear').squeeze()

        mask = (gt_t > MIN_DEPTH) & (gt_t < MAX_DEPTH) & torch.isfinite(gt_t) & (pred_t > 0)
        if mask.sum() < 10: 
            fail_reasons["Mask filtered all pixels"] += 1
            continue

        pred_t = torch.clamp(pred_t, min=MIN_DEPTH, max=MAX_DEPTH)
        metrics = compute_depth_metrics(gt_t, pred_t, mask)
        if metrics is None: continue

        results.append({
            'scene': scene_dir_name,
            'image': image_stem,
            'gt_fov': gt_fov_x,
            'pred_fov': pred_fov_x,
            'fov_diff': fov_diff,
            'fov_err': fov_err_abs,
            'rmse': metrics['rmse'],
            'abs_rel': metrics['abs_rel']
        })

    if not results:
        print("\nNo valid samples found. Debug Info:")
        # Print top failure reasons.
        for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - {reason}: {count} times")
        print("\nHint: check whether the 'Missing Scene Dir' paths match your GT root.")
        return

    # Aggregate statistics below.
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print(f"FoV Analysis Report (N={len(df)})")
    print("="*100)

    print(f"{'Metric':<20} | {'Mean':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 70)
    print(f"{'FoV Error (deg)':<20} | {df['fov_err'].mean():<10.4f} | {df['fov_err'].std():<10.4f} | {df['fov_err'].min():<10.4f} | {df['fov_err'].max():<10.4f}")
    print(f"{'RMSE (m)':<20} | {df['rmse'].mean():<10.4f} | {df['rmse'].std():<10.4f} | {df['rmse'].min():<10.4f} | {df['rmse'].max():<10.4f}")
    print(f"{'AbsRel':<20} | {df['abs_rel'].mean():<10.4f} | {df['abs_rel'].std():<10.4f} | {df['abs_rel'].min():<10.4f} | {df['abs_rel'].max():<10.4f}")
    print("-" * 100)

    corr_rmse, _ = pearsonr(df['fov_err'], df['rmse'])
    corr_absrel, _ = pearsonr(df['fov_err'], df['abs_rel'])
    
    print("\n🔗 Correlation Analysis (Pearson)")
    print(f"   Corr (FoV Err vs RMSE):   {corr_rmse:.4f}")
    print(f"   Corr (FoV Err vs AbsRel): {corr_absrel:.4f}")

    out_csv = pred_root_base / "fov_analysis_details.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()
    
    if os.path.exists(args.pred):
        run_fov_evaluation(args.pred, args.gt)
    else:
        print("Pred path not found")
