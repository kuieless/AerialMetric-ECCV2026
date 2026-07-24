import sys
import argparse
from pathlib import Path
_script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_script_dir))
from a_infer_lora96_norm import run_inference_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Inference Pipeline")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--config", required=True, help="Path to LoRA config JSON")
    parser.add_argument("--weight", required=True, help="Path to LoRA .pt weight")
    parser.add_argument("--resize", type=int, default=1024, help="Resize size; 0 means original resolution")
    parser.add_argument("--ratio", type=float, default=1.0, help="Sampling ratio; 1.0 means full dataset")
    parser.add_argument("--batch_size", type=int, required=True, help="Inference batch size; adjust based on GPU memory")
    parser.add_argument("--lora_rank", type=int, choices=[64, 96, 128], required=True,
                        help="LoRA rank (r); alpha is set to 2 * rank")
    parser.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="none",
                        help="auto: use meta.json if present; load: require meta.json; none: do not pass fov_x")
    parser.add_argument("--apply_mask", type=int, choices=[0, 1], default=1,
                        help="Apply model predicted mask to depth output (1=True, 0=False). Default 1.")

    args = parser.parse_args()

    input_roots = [args.input]
    output_root = args.output
    real_resize = args.resize if args.resize > 0 else None

    print(
        f"Starting LoRA inference "
        f"(Batch Size: {args.batch_size}, Rank: {args.lora_rank}, "
        f"Alpha: {2 * args.lora_rank}, Intrinsics: {args.intrinsics_mode})"
    )
    
    run_inference_pipeline(
        input_roots=input_roots,
        output_root=output_root,
        lora_config=args.config,
        lora_weight=args.weight,
        sampling_ratio=args.ratio,
        resize=real_resize,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        intrinsics_mode=args.intrinsics_mode,
        apply_mask=bool(args.apply_mask),
    )
