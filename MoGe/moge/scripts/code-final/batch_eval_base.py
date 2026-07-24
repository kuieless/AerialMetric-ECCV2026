"""
Batch evaluation script for non-LoRA MoGe checkpoints.
Supports full-parameter, head-only, and neck-only checkpoints through the same
inference/extraction/evaluation pipeline.
"""
import os
import re
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def get_sorted_checkpoints(root_dir, step_interval=1000, exact_steps=None,
                           min_step=0, max_step=999999, ignore_latest=True,
                           include_named=True):
    """Scan checkpoint directory and return filtered .pt paths."""
    if not os.path.exists(root_dir):
        print(f"ERROR: Checkpoint directory not found: {root_dir}")
        return []

    if exact_steps is None:
        exact_steps = []

    files = glob.glob(os.path.join(root_dir, "*.pt"))
    selected = []

    def extract_step(path):
        stem = Path(path).stem
        if "latest" in stem:
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
    cmd_str = " ".join(cmd)
    script_name = os.path.basename(cmd[1]) if len(cmd) > 1 else cmd[0]
    print(f"    Running: {script_name} ... (log: {os.path.basename(log_file)})")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {cmd_str}\n")
        subprocess.run(cmd, check=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


def build_active_datasets(args):
    active = {}
    if args.decoupled_input:
        active["Decoupled"] = {"input": args.decoupled_input, "gt": args.decoupled_gt}
    if args.oblique_input:
        active["Oblique"] = {"input": args.oblique_input, "gt": args.oblique_gt}
    if args.wild_input:
        active["Wild"] = {"input": args.wild_input, "gt": args.wild_gt}
    return active


def process_checkpoint(ckpt_path, args, env):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n  [Processing {args.model_type}] {ckpt_name}")

    base_dir = os.path.join(args.output_dir, args.model_type, ckpt_name)
    os.makedirs(base_dir, exist_ok=True)
    infer_out = os.path.join(base_dir, "Infer")
    extract_out = os.path.join(base_dir, "Extracted")
    run_log = os.path.join(base_dir, "run.log")

    active = build_active_datasets(args)
    if not active:
        print("    No datasets enabled, skipping.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: inference
    for ds_name, ds_cfg in active.items():
        expected_out = os.path.join(infer_out, ds_name)
        has_files = os.path.exists(expected_out) and len(list(Path(expected_out).rglob("*.npy"))) > 0
        if has_files:
            print(f"    Skip inference: {ds_name} (results exist)")
            continue

        try:
            run_cmd([
                "python", "a-infer_norm.py",
                "--input", ds_cfg["input"],
                "--output", expected_out,
                "--model", ckpt_path,
                "--ratio", str(args.sampling_ratio),
                "--resize", str(args.resize),
                "--batch_size", str(args.batch_size),
                "--intrinsics_mode", args.intrinsics_mode,
            ], log_file=run_log, cwd=script_dir, env=env)
        except Exception as e:
            print(f"    ERROR inference {ds_name}: {e}")

    # Step 2: extraction
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

    # Step 3: evaluation
    if args.decoupled_input and args.decoupled_gt:
        decoupled_pred = os.path.join(extract_out, "Decoupled")
        report = os.path.join(decoupled_pred, "Eval_Report_Decoupled.txt")
        if not os.path.exists(report) and os.path.exists(decoupled_pred):
            try:
                run_cmd(["python", "c-eval-bench.py", "--pred", decoupled_pred, "--gt", args.decoupled_gt],
                        log_file=run_log, cwd=script_dir, env=env)
            except Exception as e:
                print(f"    ERROR decoupled eval: {e}")

    if args.oblique_input and args.oblique_gt:
        oblique_pred = os.path.join(extract_out, "Oblique")
        report = os.path.join(oblique_pred, "Eval_Report_Oblique_Pixel.txt")
        if not os.path.exists(report) and os.path.exists(oblique_pred):
            try:
                run_cmd(["python", "c-eval-oblique.py", "--pred", oblique_pred, "--gt", args.oblique_gt],
                        log_file=run_log, cwd=script_dir, env=env)
            except Exception as e:
                print(f"    ERROR oblique eval: {e}")

    if args.wild_input and args.wild_gt:
        wild_pred = os.path.join(extract_out, "Wild")
        if os.path.exists(wild_pred):
            report_multi = os.path.join(wild_pred, "Eval_Report_Wild_MultiRange.txt")
            if not os.path.exists(report_multi):
                try:
                    run_cmd(["python", "c-eval-wild.py", "--pred", wild_pred, "--gt", args.wild_gt],
                            log_file=run_log, cwd=script_dir, env=env)
                except Exception as e:
                    print(f"    ERROR wild eval: {e}")

            report_fov = os.path.join(wild_pred, "fov_analysis_details.csv")
            if not os.path.exists(report_fov):
                try:
                    run_cmd(["python", "c-eval-wild-fov.py", "--pred", wild_pred, "--gt", args.wild_gt],
                            log_file=run_log, cwd=script_dir, env=env)
                except Exception as e:
                    print(f"    ERROR wild fov eval: {e}")


def parse_exact_steps(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Non-LoRA MoGe Batch Evaluation")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", default="", help="Single .pt checkpoint")
    ckpt_group.add_argument("--checkpoint_root", default="", help="Directory containing .pt checkpoints")

    parser.add_argument("--model_type", choices=["full", "head", "neck", "baseline"], required=True,
                        help="Label used in output directory; all types use the same base inference path")
    parser.add_argument("--decoupled_input", default="", help="Decoupled inference input, e.g. decoupled")
    parser.add_argument("--decoupled_gt", default="", help="Decoupled ground-truth root")
    parser.add_argument("--oblique_input", default="", help="Oblique inference input, e.g. Oblique or Oblique-norm")
    parser.add_argument("--oblique_gt", default="", help="Oblique ground-truth root")
    parser.add_argument("--wild_input", default="", help="Wild inference input")
    parser.add_argument("--wild_gt", default="", help="Wild ground-truth root")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--step_interval", type=int, default=1000, help="Evaluate every N steps when using --checkpoint_root")
    parser.add_argument("--exact_steps", default="", help="Comma-separated extra steps to include")
    parser.add_argument("--sampling_ratio", type=float, default=1.0, help="Data sampling ratio")
    parser.add_argument("--resize", type=int, default=0, help="Resize (0=original size)")
    parser.add_argument("--batch_size", type=int, required=True, help="Inference batch size")
    parser.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="auto",
                        help="auto: use meta.json if present; load: require meta.json; none: do not pass fov_x")
    parser.add_argument("--exclude_named_checkpoints", action="store_true",
                        help="Ignore .pt files without numeric steps")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = get_sorted_checkpoints(
            args.checkpoint_root,
            step_interval=args.step_interval,
            exact_steps=parse_exact_steps(args.exact_steps),
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
