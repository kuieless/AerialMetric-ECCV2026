"""
Batch evaluation wrapper for LoRA ground depth estimation.
Runs the dedicated LoRA ground evaluator without touching the baseline path.
"""
import argparse
import glob
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


def get_sorted_checkpoints(
    root_dir,
    step_interval=1000,
    exact_steps=None,
    min_step=0,
    max_step=999999,
    ignore_latest=True,
    include_named=True,
):
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
        match = re.search(r"(?:^|[_-])(\d{4,})(?:$|[_-])", stem)
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


def process_checkpoint(ckpt_path, args, env):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n  [Processing LoRA] {ckpt_name}")

    base_dir = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(base_dir, exist_ok=True)
    output_json = os.path.join(base_dir, "metrics.json")
    run_log = os.path.join(base_dir, "run.log")

    if os.path.exists(output_json):
        print("    Skip evaluation: results exist")
        return

    script_dir = Path(__file__).resolve().parents[3]
    cmd = [
        "python",
        "moge/scripts/eval_baselinelora.py",
        "--lora_config",
        args.lora_config,
        "--lora_weight",
        ckpt_path,
        "--config",
        args.config,
        "--output",
        output_json,
        "--ratio",
        str(args.sampling_ratio),
        "--batch_size",
        str(args.batch_size),
        "--lora_rank",
        str(args.lora_rank),
        "--oracle" if args.oracle else "",
    ]
    cmd = [x for x in cmd if x]

    if args.dump_pred:
        cmd.append("--dump_pred")
    if args.dump_gt:
        cmd.append("--dump_gt")

    try:
        run_cmd(cmd, log_file=run_log, cwd=script_dir, env=env)
        print(f"    Done: {ckpt_name}")
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: evaluation failed (exit code {e.returncode})")
    except Exception as e:
        print(f"    ERROR: {e}")


def parse_exact_steps(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="LoRA Ground Batch Evaluation")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", default="", help="Single .pt checkpoint")
    ckpt_group.add_argument("--checkpoint_root", default="", help="Directory containing .pt checkpoints")

    parser.add_argument("--lora_config", required=True, help="Path to the LoRA train config JSON")
    parser.add_argument("--config", required=True, help="Path to the ground benchmark config JSON")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--step_interval", type=int, default=1000, help="Evaluate every N steps when using --checkpoint_root")
    parser.add_argument("--exact_steps", default="", help="Comma-separated extra steps to include")
    parser.add_argument("--sampling_ratio", type=float, default=1.0, help="Data sampling ratio")
    parser.add_argument("--batch_size", type=int, required=True, help="Inference batch size")
    parser.add_argument("--lora_rank", type=int, choices=[64, 96, 128], required=True, help="LoRA rank")
    parser.add_argument("--oracle", action="store_true", help="Use GT intrinsics when available")
    parser.add_argument("--dump_pred", action="store_true", help="Dump prediction artifacts")
    parser.add_argument("--dump_gt", action="store_true", help="Dump ground-truth artifacts")
    parser.add_argument("--exclude_named_checkpoints", action="store_true", help="Ignore .pt files without numeric steps")
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
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(checkpoints)}] {Path(ckpt).stem}")
        print(f"{'=' * 60}")
        process_checkpoint(ckpt, args, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
