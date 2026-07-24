"""
Batch evaluation script for LoRA aerial depth estimation.
Runs inference -> extraction -> evaluation across Decoupled, Oblique, and Wild datasets.

Usage:
    python batch_eval.py \
        --checkpoint_root /path/to/checkpoints \
        --lora_config configs/Final_train/config-lora-all.json \
        --decoupled_path /path/to/decoupled \
        --oblique_path /path/to/Oblique \
        --wild_path /path/to/Wild \
        --output_dir /path/to/results \
        --gpu 0
"""
import os
import sys
import re
import glob
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def get_sorted_checkpoints(root_dir, step_interval=1000, exact_steps=None,
                           min_step=0, max_step=999999, ignore_latest=True,
                           include_named=True):
    """Scan checkpoint directory and return filtered, sorted .pt paths."""
    if not os.path.exists(root_dir):
        print(f"ERROR: Checkpoint directory not found: {root_dir}")
        return []

    if exact_steps is None:
        exact_steps = []

    files = glob.glob(os.path.join(root_dir, "*.pt"))
    selected = []

    def extract_step(path):
        stem = Path(path).stem
        if stem == "latest":
            return -1
        if stem.isdigit():
            return int(stem)
        match = re.search(r'(?:^|[_-])(\d{4,})(?:$|[_-])', stem)
        return int(match.group(1)) if match else -2

    for f in files:
        step = extract_step(f)
        if step == -2:
            if include_named:
                selected.append(f)
            continue
        if step == -1:
            if not ignore_latest:
                selected.append(f)
            continue

        if step in exact_steps:
            selected.append(f)
        elif min_step <= step <= max_step:
            if step_interval and step_interval > 0:
                if step % step_interval == 0:
                    selected.append(f)
            else:
                selected.append(f)

    selected = list(set(selected))
    selected.sort(key=lambda x: (extract_step(x) < 0, extract_step(x), Path(x).stem))
    return selected


def run_cmd(cmd, log_file, cwd=None, env=None):
    """Execute a command, redirecting output to log file."""
    cmd_str = " ".join(cmd)
    script_name = os.path.basename(cmd[1]) if len(cmd) > 1 else cmd[0]
    print(f"    Running: {script_name} ... (log: {os.path.basename(log_file)})")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {cmd_str}\n")
        subprocess.run(cmd, check=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


def process_checkpoint(ckpt_path, args, env):
    """Run full pipeline for a single checkpoint on Decoupled, Oblique, Wild."""
    ckpt_name = Path(ckpt_path).stem
    print(f"\n  [Processing] {ckpt_name}")

    base_dir = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(base_dir, exist_ok=True)
    infer_out = os.path.join(base_dir, "Infer")
    extract_out = os.path.join(base_dir, "Extracted")
    run_log = os.path.join(base_dir, "run.log")

    # Determine which datasets to process
    active = {}
    if args.decoupled_path:
        active["Decoupled"] = args.decoupled_path
    if args.oblique_path:
        active["Oblique"] = args.oblique_path
    if args.wild_path:
        active["Wild"] = args.wild_path

    if not active:
        print("    No datasets enabled, skipping.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Step 1: Inference ---
    for ds_name, ds_path in active.items():
        expected_out = os.path.join(infer_out, ds_name)
        has_files = os.path.exists(expected_out) and len(list(Path(expected_out).rglob("*.npy"))) > 0
        if has_files:
            print(f"    Skip inference: {ds_name} (results exist)")
            continue

        actual_input = ds_path
        temp_dir = None

        # Wild dataset: filter blacklisted scenes via symlinks
        if ds_name == "Wild" and args.wild_exclude:
            temp_dir = os.path.join(infer_out, ".Wild_Filtered")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"    Filtering Wild: excluding {len(args.wild_exclude)} scenes...")
            for scene in os.listdir(ds_path):
                if scene in args.wild_exclude or not os.path.isdir(os.path.join(ds_path, scene)):
                    continue
                src_img_dir = os.path.join(ds_path, scene, "image")
                if not os.path.exists(src_img_dir):
                    continue
                dst_img_dir = os.path.join(temp_dir, scene, "image")
                os.makedirs(dst_img_dir, exist_ok=True)
                for img_file in os.listdir(src_img_dir):
                    src = os.path.join(src_img_dir, img_file)
                    dst = os.path.join(dst_img_dir, img_file)
                    if not os.path.exists(dst) and os.path.isfile(src):
                        os.symlink(src, dst)
            actual_input = temp_dir

        try:
            run_cmd([
                "python", "a-infer_lora96-norm.py",
                "--input", actual_input,
                "--output", expected_out,
                "--config", args.lora_config,
                "--weight", ckpt_path,
                "--ratio", str(args.sampling_ratio),
                "--resize", str(args.resize),
                "--batch_size", str(args.batch_size),
                "--lora_rank", str(args.lora_rank),
                "--intrinsics_mode", args.intrinsics_mode,
            ], log_file=run_log, cwd=script_dir, env=env)
        except Exception as e:
            print(f"    ERROR inference {ds_name}: {e}")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    # --- Step 2: Extraction ---
    need_extract = not os.path.exists(extract_out) or len(os.listdir(extract_out)) == 0
    if not need_extract:
        print("    Skip extraction (results exist)")
    else:
        try:
            run_cmd([
                "python", "b-extra.py",
                "--input", infer_out,
                "--output", extract_out,
                "--target", "depth.npy",
                "--ext", ".npy",
            ], log_file=run_log, cwd=script_dir, env=env)
        except Exception as e:
            print(f"    ERROR extraction: {e}")

    # --- Step 3: Evaluation ---
    if args.decoupled_path:
        decoupled_pred = os.path.join(extract_out, "Decoupled")
        report = os.path.join(decoupled_pred, "Eval_Report_Decoupled.txt")
        if not os.path.exists(report) and os.path.exists(decoupled_pred):
            try:
                run_cmd(["python", "c-eval-bench.py", "--pred", decoupled_pred, "--gt", args.decoupled_path],
                        log_file=run_log, cwd=script_dir, env=env)
            except Exception as e:
                print(f"    ERROR decoupled eval: {e}")

    if args.oblique_path:
        oblique_pred = os.path.join(extract_out, "Oblique")
        report = os.path.join(oblique_pred, "Eval_Report_Oblique_Pixel.txt")
        if not os.path.exists(report) and os.path.exists(oblique_pred):
            try:
                run_cmd(["python", "c-eval-oblique.py", "--pred", oblique_pred, "--gt", args.oblique_path],
                        log_file=run_log, cwd=script_dir, env=env)
            except Exception as e:
                print(f"    ERROR oblique eval: {e}")

    if args.wild_path:
        wild_pred = os.path.join(extract_out, "Wild")
        if os.path.exists(wild_pred):
            report_multi = os.path.join(wild_pred, "Eval_Report_Wild_MultiRange.txt")
            if not os.path.exists(report_multi):
                try:
                    run_cmd(["python", "c-eval-wild.py", "--pred", wild_pred, "--gt", args.wild_path],
                            log_file=run_log, cwd=script_dir, env=env)
                except Exception as e:
                    print(f"    ERROR wild eval: {e}")

            report_fov = os.path.join(wild_pred, "fov_analysis_details.csv")
            if not os.path.exists(report_fov):
                try:
                    run_cmd(["python", "c-eval-wild-fov.py", "--pred", wild_pred, "--gt", args.wild_path],
                            log_file=run_log, cwd=script_dir, env=env)
                except Exception as e:
                    print(f"    ERROR wild fov eval: {e}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Aerial Batch Evaluation")
    parser.add_argument("--checkpoint_root", required=True, help="Directory containing .pt checkpoints")
    parser.add_argument("--lora_config", required=True, help="Path to LoRA config JSON")
    parser.add_argument("--decoupled_path", default="", help="Path to Decoupled dataset")
    parser.add_argument("--oblique_path", default="", help="Path to Oblique dataset")
    parser.add_argument("--wild_path", default="", help="Path to Wild dataset")
    parser.add_argument("--wild_exclude", nargs="*", default=[], help="Wild scene names to exclude")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--step_interval", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--sampling_ratio", type=float, default=1.0, help="Data sampling ratio")
    parser.add_argument("--resize", type=int, default=0, help="Resize (0=original size)")
    parser.add_argument("--batch_size", type=int, required=True, help="Inference batch size")
    parser.add_argument("--lora_rank", type=int, choices=[64, 96, 128], required=True,
                        help="LoRA rank (r); alpha is set to 2 * rank")
    parser.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="auto",
                        help="auto: use meta.json if present; load: require meta.json; none: do not pass fov_x")
    parser.add_argument("--exclude_named_checkpoints", action="store_true",
                        help="Ignore .pt files without numeric steps, such as Moge2-Aerial.pt")
    args = parser.parse_args()

    checkpoints = get_sorted_checkpoints(
        args.checkpoint_root,
        step_interval=args.step_interval,
        include_named=not args.exclude_named_checkpoints,
    )
    print(f"Found {len(checkpoints)} checkpoints: {[Path(p).stem for p in checkpoints]}")

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for i, ckpt in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(checkpoints)}] {Path(ckpt).stem}")
        print(f"{'='*60}")
        process_checkpoint(ckpt, args, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
