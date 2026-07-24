"""
Unified ground benchmark entry point for third-party depth baselines.

This wrapper calls Ground_MoGe/moge/scripts/eval_baseline.py with the
corresponding baseline adapter and local third-party source tree.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


BASELINE_BY_MODEL = {
    "unidepth": "unidepth.py",
    "depthpro": "ml_depth_pro.py",
    "zoedepth": "zoedepth.py",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_command(args):
    root = repo_root()
    ground_root = root / "Ground_MoGe"
    baseline_path = ground_root / "baselines" / BASELINE_BY_MODEL[args.model]
    eval_script = ground_root / "moge" / "scripts" / "eval_baseline.py"
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not config_path.is_absolute():
        config_path = root / config_path
    if not output_path.is_absolute():
        output_path = root / output_path

    cmd = [
        sys.executable,
        str(eval_script),
        "--baseline",
        str(baseline_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--log_interval",
        str(args.log_interval),
        "--log_level",
        args.log_level,
    ]

    if args.oracle:
        cmd.append("--oracle")
    if args.all_metrics:
        cmd.append("--all_metrics")

    if args.model == "unidepth":
        cmd.extend([
            "--repo",
            str(root / "Depth_Baselines" / "UniDepth"),
            "--version",
            args.unidepth_version,
            "--backbone",
            args.unidepth_backbone,
            "--device",
            args.device,
        ])
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.pretrained:
            cmd.extend(["--pretrained", args.pretrained])
        if not args.use_intrinsics:
            cmd.append("--no_use_intrinsics")
        if args.fp16:
            cmd.append("--fp16")
        return cmd

    if args.model == "depthpro":
        cmd.extend([
            "--repo",
            str(root / "Depth_Baselines" / "ml-depth-pro"),
            "--device",
            args.device,
            "--default_focal_px",
            str(args.default_focal_px),
        ])
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.fp16:
            cmd.append("--fp16")
        return cmd

    if args.model == "zoedepth":
        cmd.extend([
            "--repo",
            str(root / "Depth_Baselines" / "ZoeDepth"),
            "--variant",
            args.zoedepth_variant,
            "--device",
            args.device,
        ])
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.pretrained_resource:
            cmd.extend(["--pretrained_resource", args.pretrained_resource])
        if not args.pad_input:
            cmd.append("--no_pad_input")
        if not args.flip_aug:
            cmd.append("--no_flip_aug")
        return cmd

    raise ValueError(f"Unsupported model: {args.model}")


def default_output(model: str, oracle: bool) -> str:
    suffix = "_gt_intr" if oracle else ""
    return f"eval_output_release/{model}_ground{suffix}.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate third-party depth baselines on the ground benchmark.")
    parser.add_argument("--model", required=True, choices=sorted(BASELINE_BY_MODEL), help="Baseline to evaluate.")
    parser.add_argument(
        "--config",
        default="Ground_MoGe/configs/eval/ground_metric_benchmarks_local.json",
        help="Ground benchmark config path.",
    )
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to eval_output_release/<model>_ground.json.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--device", default="cuda", help="Device passed to the baseline adapter.")
    parser.add_argument("--checkpoint", default="", help="Optional local checkpoint path.")
    parser.add_argument("--oracle", action="store_true", help="Use GT intrinsics in evaluation.")
    parser.add_argument("--all_metrics", action="store_true", help="Compute all metrics instead of metric-depth-only.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 where supported by the baseline.")
    parser.add_argument("--log_interval", type=int, default=50, help="Print running metrics every N samples.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")

    parser.add_argument("--unidepth_version", default="v2", choices=["v1", "v2"], help="UniDepth version.")
    parser.add_argument("--unidepth_backbone", default="vitl14", choices=["vitl14", "vitb14", "vits14", "cnvnxtl"], help="UniDepth backbone.")
    parser.add_argument("--pretrained", default="", help="Optional UniDepth HuggingFace model id or local model dir.")
    parser.add_argument("--use_intrinsics", action=argparse.BooleanOptionalAction, default=True, help="Pass GT intrinsics to UniDepth when available.")

    parser.add_argument("--default_focal_px", type=float, default=0.0, help="DepthPro fallback focal length. <=0 lets DepthPro estimate focal length.")

    parser.add_argument("--zoedepth_variant", default="nk", choices=["nk", "n", "k"], help="ZoeDepth variant.")
    parser.add_argument("--pretrained_resource", default="", help="Optional ZoeDepth pretrained_resource string.")
    parser.add_argument("--pad_input", action=argparse.BooleanOptionalAction, default=True, help="Enable ZoeDepth input padding.")
    parser.add_argument("--flip_aug", action=argparse.BooleanOptionalAction, default=True, help="Enable ZoeDepth flip augmentation.")

    args = parser.parse_args()
    if not args.output:
        args.output = default_output(args.model, args.oracle)
    return args


def main():
    args = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = build_command(args)
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root(), env=env)


if __name__ == "__main__":
    main()
