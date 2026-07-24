"""
Unified inference/evaluation CLI for AerialMetric/MoGe variants.

Supported model types:
  - full, head, neck: non-LoRA checkpoints loaded by a-infer_norm.py
  - lora64, lora96, lora128: LoRA checkpoints loaded by a-infer_lora96-norm.py

When --intrinsics_mode=load is used, the input roots must point to the
norm-style dataset layout that includes per-sample meta.json files.

The pipeline is:
  inference -> extraction -> dataset-specific evaluation -> optional cleanup
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


LORA_RANK_BY_TYPE = {
    "lora64": 64,
    "lora96": 96,
    "lora128": 128,
}


def parse_exact_steps(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_step(path):
    stem = Path(path).stem
    if "latest" in stem:
        return -1
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(?:^|[_-])(\d{4,})(?:$|[_-])", stem)
    return int(match.group(1)) if match else -2


def collect_checkpoints(args):
    if args.checkpoint:
        return [args.checkpoint]

    files = glob.glob(os.path.join(args.checkpoint_root, "*.pt"))
    exact_steps = set(parse_exact_steps(args.exact_steps))
    selected = []

    for path in files:
        step = checkpoint_step(path)
        if step == -2:
            if not args.exclude_named_checkpoints:
                selected.append(path)
            continue
        if step == -1:
            if args.include_latest:
                selected.append(path)
            continue
        if step in exact_steps:
            selected.append(path)
            continue
        if not (args.min_step <= step <= args.max_step):
            continue
        if args.step_interval <= 0 or step % args.step_interval == 0:
            selected.append(path)

    selected = sorted(set(selected), key=lambda p: (checkpoint_step(p) < 0, checkpoint_step(p), Path(p).stem))
    return selected


def run_cmd(cmd, log_file, cwd, env):
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"    Running: {Path(cmd[1]).name if len(cmd) > 1 else cmd[0]}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {cmd_str}\n")
        subprocess.run(cmd, check=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


def active_datasets(args):
    datasets = {}
    if args.decoupled_input:
        datasets["Decoupled"] = {
            "input": args.decoupled_input,
            "gt": args.decoupled_gt,
            "csv_dir": args.decoupled_csv_dir,
            "mask_dir": args.decoupled_mask_dir,
        }
    if args.oblique_input:
        datasets["Oblique"] = {
            "input": args.oblique_input,
            "gt": args.oblique_gt,
            "mask_dir": args.oblique_mask_dir,
        }
    if args.wild_input:
        datasets["Wild"] = {
            "input": args.wild_input,
            "gt": args.wild_gt,
        }
    return datasets


def root_has_meta_json(root):
    if not root:
        return False
    try:
        next(Path(root).rglob("meta.json"))
        return True
    except StopIteration:
        return False


def dataset_intrinsics_mode(args, ds_name):
    if ds_name == "Wild" and args.intrinsics_mode == "load":
        return "auto"
    return args.intrinsics_mode


def validate_args(args):
    if args.model_type in LORA_RANK_BY_TYPE and not args.lora_config:
        raise SystemExit("--lora_config is required for lora64/lora96/lora128")

    datasets = active_datasets(args)
    if not datasets:
        raise SystemExit("At least one dataset input is required: --decoupled_input, --oblique_input, or --wild_input")

    for name, ds in datasets.items():
        if not ds["gt"]:
            raise SystemExit(f"--{name.lower()}_gt is required when --{name.lower()}_input is set")
        if args.mask_mode == "load" and name in {"Decoupled", "Oblique"} and not ds.get("mask_dir"):
            raise SystemExit(f"--{name.lower()}_mask_dir is required when --mask_mode load is used")
        if name in {"Decoupled", "Oblique"} and dataset_intrinsics_mode(args, name) == "load" and not root_has_meta_json(ds["input"]):
            raise SystemExit(
                f"{name} input root has no meta.json files. "
                f"Switch to the norm-style {name.lower()} dataset layout before using --intrinsics_mode load."
            )


def inference_command(args, ds_name, ckpt_path, dataset_input, dataset_output):
    python = sys.executable
    intrinsics_mode = dataset_intrinsics_mode(args, ds_name)
    # Wild uses apply_mask=True (old behavior), others use apply_mask=False
    apply_mask = 0 if ds_name in ("Decoupled", "Oblique") else 1
    if args.model_type in LORA_RANK_BY_TYPE:
        return [
            python, "a-infer_lora96-norm.py",
            "--input", dataset_input,
            "--output", dataset_output,
            "--config", args.lora_config,
            "--weight", ckpt_path,
            "--ratio", str(args.sampling_ratio),
            "--resize", str(args.resize),
            "--batch_size", str(args.batch_size),
            "--lora_rank", str(LORA_RANK_BY_TYPE[args.model_type]),
            "--intrinsics_mode", intrinsics_mode,
            "--apply_mask", str(apply_mask),
        ]

    return [
        python, "a-infer_norm.py",
        "--input", dataset_input,
        "--output", dataset_output,
        "--model", ckpt_path,
        "--ratio", str(args.sampling_ratio),
        "--resize", str(args.resize),
        "--batch_size", str(args.batch_size),
        "--intrinsics_mode", intrinsics_mode,
        "--apply_mask", str(apply_mask),
    ]


def evaluate_dataset(args, ds_name, ds_cfg, infer_out, extract_out, script_dir, log_file, env):
    pred_dir = os.path.join(extract_out, ds_name)
    if not os.path.exists(pred_dir):
        print(f"    Skip eval: {ds_name} predictions not found")
        return

    python = sys.executable
    if ds_name == "Decoupled":
        report = os.path.join(pred_dir, "Eval_Report_Decoupled.txt")
        if os.path.exists(report):
            print("    Skip eval: Decoupled report exists")
            return
        cmd = [python, "c-eval-bench.py", "--pred", pred_dir, "--gt", ds_cfg["gt"]]
        if ds_cfg.get("csv_dir"):
            cmd.extend(["--csv_dir", ds_cfg["csv_dir"]])
        if args.mask_mode == "load" and ds_cfg.get("mask_dir"):
            cmd.extend(["--mask_dir", ds_cfg["mask_dir"]])
        run_cmd(cmd, log_file, script_dir, env)
        return

    if ds_name == "Oblique":
        report = os.path.join(pred_dir, "Eval_Report_Oblique_Pixel.txt")
        if os.path.exists(report):
            print("    Skip eval: Oblique report exists")
            return
        cmd = [python, "c-eval-oblique.py", "--pred", pred_dir, "--gt", ds_cfg["gt"]]
        if args.mask_mode == "load" and ds_cfg.get("mask_dir"):
            cmd.extend(["--mask_dir", ds_cfg["mask_dir"]])
        run_cmd(cmd, log_file, script_dir, env)
        return

    if ds_name == "Wild":
        report_multi = os.path.join(pred_dir, "Eval_Report_Wild_MultiRange.txt")
        if not os.path.exists(report_multi):
            run_cmd([python, "c-eval-wild.py", "--pred", pred_dir, "--gt", ds_cfg["gt"]], log_file, script_dir, env)
        if not args.eval_wild_fov:
            return

        # FoV evaluation needs the unflattened inference layout because each
        # sample directory contains both depth.npy and fov.json.
        fov_pred_dir = os.path.join(infer_out, ds_name)
        report_fov = os.path.join(pred_dir, "fov_analysis_details.csv")
        source_report_fov = os.path.join(fov_pred_dir, "fov_analysis_details.csv")
        if not os.path.exists(report_fov):
            if not os.path.exists(source_report_fov):
                run_cmd([python, "c-eval-wild-fov.py", "--pred", fov_pred_dir, "--gt", ds_cfg["gt"]], log_file, script_dir, env)
            if os.path.exists(source_report_fov):
                shutil.copy2(source_report_fov, report_fov)


def cleanup_intermediate(base_dir):
    infer_dir = Path(base_dir) / "Infer"
    extract_dir = Path(base_dir) / "Extracted"

    if infer_dir.exists():
        shutil.rmtree(infer_dir)

    removed_npy = 0
    if extract_dir.exists():
        for path in extract_dir.rglob("*.npy"):
            path.unlink()
            removed_npy += 1

    print(f"    Cleanup: removed Infer={infer_dir}, extracted_npy={removed_npy}")


def reports_complete(args, datasets, extract_out):
    expected = []
    for ds_name in datasets:
        pred_dir = Path(extract_out) / ds_name
        if ds_name == "Decoupled":
            expected.append(pred_dir / "Eval_Report_Decoupled.txt")
        elif ds_name == "Oblique":
            expected.append(pred_dir / "Eval_Report_Oblique_Pixel.txt")
        elif ds_name == "Wild":
            expected.append(pred_dir / "Eval_Report_Wild_MultiRange.txt")
            if args.eval_wild_fov:
                expected.append(pred_dir / "fov_analysis_details.csv")
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        print("    Cleanup skipped: missing report(s)")
        for path in missing:
            print(f"      - {path}")
        return False
    return True


def process_checkpoint(args, ckpt_path, script_dir, env):
    ckpt_name = Path(ckpt_path).stem
    model_dir = Path(args.output_dir) / args.model_type / ckpt_name
    infer_out = model_dir / "Infer"
    extract_out = model_dir / "Extracted"
    log_file = model_dir / "run.log"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{args.model_type}] {ckpt_name}")
    datasets = active_datasets(args)
    print(f"    Active datasets: {', '.join(datasets)}")

    for ds_name, ds_cfg in datasets.items():
        dataset_output = infer_out / ds_name
        has_depth = dataset_output.exists() and any(dataset_output.rglob("depth.npy"))
        if has_depth and not args.force:
            print(f"    Skip inference: {ds_name} exists")
            continue
        if has_depth and args.force:
            shutil.rmtree(dataset_output)

        run_cmd(
            inference_command(args, ds_name, ckpt_path, ds_cfg["input"], str(dataset_output)),
            str(log_file),
            script_dir,
            env,
        )

    if extract_out.exists() and any(extract_out.rglob("*.npy")) and not args.force:
        print("    Skip extraction: extracted npy exists")
    else:
        if extract_out.exists() and args.force:
            shutil.rmtree(extract_out)
        run_cmd(
            [sys.executable, "b-extra.py", "--input", str(infer_out), "--output", str(extract_out), "--target", "depth.npy", "--ext", ".npy"],
            str(log_file),
            script_dir,
            env,
        )

    for ds_name, ds_cfg in datasets.items():
        evaluate_dataset(args, ds_name, ds_cfg, str(infer_out), str(extract_out), script_dir, str(log_file), env)

    if args.cleanup_intermediate and reports_complete(args, datasets, extract_out):
        cleanup_intermediate(model_dir)


def main():
    parser = argparse.ArgumentParser(description="Unified AerialMetric inference/evaluation CLI")
    parser.add_argument("--model_type", required=True,
                        choices=["full", "head", "neck", "lora64", "lora96", "lora128"],
                        help="Evaluation variant. LoRA rank/alpha are inferred from lora64/lora96/lora128.")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", default="", help="Single checkpoint path")
    ckpt_group.add_argument("--checkpoint_root", default="", help="Directory containing .pt checkpoints")

    parser.add_argument("--lora_config", default="", help="LoRA training config JSON; required for LoRA variants")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")

    parser.add_argument("--decoupled_input", default="", help="Decoupled input root. Use decoupled for auto, or the norm-style decoupled-norm root for load.")
    parser.add_argument("--decoupled_gt", default="", help="Decoupled GT root, e.g. /data1/szq/Val/decoupled-norm")
    parser.add_argument("--decoupled_csv_dir", default="", help="Decoupled CSV metadata root, e.g. decoupled")
    parser.add_argument("--bench_input", dest="decoupled_input", help=argparse.SUPPRESS)
    parser.add_argument("--bench_gt", dest="decoupled_gt", help=argparse.SUPPRESS)
    parser.add_argument("--bench_csv_dir", dest="decoupled_csv_dir", help=argparse.SUPPRESS)
    parser.add_argument("--oblique_input", default="", help="Oblique input root. Use Oblique for auto, or Oblique-norm for load.")
    parser.add_argument("--oblique_gt", default="", help="Oblique GT root")
    parser.add_argument("--oblique_mask_dir", default="", help="Oblique mask root, e.g. /data1/szq/Val/Oblique-masks")
    parser.add_argument("--wild_input", default="", help="Wild input root")
    parser.add_argument("--wild_gt", default="", help="Wild GT root")
    parser.add_argument("--decoupled_mask_dir", default="", help="Decoupled mask root, e.g. /data1/szq/Val/decoupled-masks")

    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size; use 8 to reproduce old LoRA runs")
    parser.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="none",
                        help="auto: use meta.json if present; load: require meta.json and norm-style input roots; none: do not pass fov_x")
    parser.add_argument("--mask_mode", choices=["none", "load"], default="none",
                        help="Load per-sample PNG masks for Decoupled/Oblique and exclude white pixels from evaluation.")
    parser.add_argument("--resize", type=int, default=0, help="Resize long edge; 0 means original size")
    parser.add_argument("--sampling_ratio", type=float, default=1.0, help="Data sampling ratio")
    parser.add_argument("--eval_wild_fov", action="store_true",
                        help="Optional internal analysis: evaluate Wild FoV predictions and write fov_analysis_details.csv")

    parser.add_argument("--step_interval", type=int, default=1000, help="Evaluate every N steps when using checkpoint_root")
    parser.add_argument("--exact_steps", default="", help="Comma-separated extra checkpoint steps")
    parser.add_argument("--min_step", type=int, default=0, help="Minimum checkpoint step")
    parser.add_argument("--max_step", type=int, default=999999999, help="Maximum checkpoint step")
    parser.add_argument("--include_latest", action="store_true", help="Include latest*.pt checkpoints")
    parser.add_argument("--exclude_named_checkpoints", action="store_true",
                        help="Ignore checkpoints without a numeric step, e.g. Moge2-Aerial.pt")

    parser.add_argument("--force", action="store_true", help="Overwrite existing inference/extraction outputs")
    parser.add_argument("--cleanup_intermediate", action="store_true",
                        help="After successful eval, delete Infer and extracted .npy files, keeping reports/logs")
    args = parser.parse_args()

    validate_args(args)
    checkpoints = collect_checkpoints(args)
    print(f"Found {len(checkpoints)} checkpoint(s): {[Path(p).stem for p in checkpoints]}")
    if not checkpoints:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, ckpt_path in enumerate(checkpoints, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(checkpoints)}] {Path(ckpt_path).name}")
        print(f"{'=' * 80}")
        process_checkpoint(args, ckpt_path, script_dir, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
