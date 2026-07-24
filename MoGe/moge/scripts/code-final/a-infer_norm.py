import sys
from pathlib import Path

# Ensure we can import from the parent scripts directory
_script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_script_dir))

from infer_core_norm import run_base_inference_pipeline

# User defaults

EXP_NAME = "Base_Model_V2_Original_Size"

# Input dataset defaults.
INPUT_ROOTS = []
OUTPUT_ROOT = ""
MODEL_PATH = ""
PARAMS = {
    "version": "v2",
    "sampling_ratio": 1.0,
    "resize": None,
    "device": "cuda"
}

# Entry point

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Model Inference Pipeline")
    parser.add_argument("--input", default=None, help="Input root directory")
    parser.add_argument("--output", default=None, help="Output root directory")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    
    parser.add_argument("--resize", type=int, default=0, help="Resize size; 0 means original resolution")
    parser.add_argument("--ratio", type=float, default=1.0, help="Sampling ratio")
    parser.add_argument("--batch_size", type=int, required=True, help="Inference batch size; adjust based on GPU memory")
    parser.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="none",
                        help="auto: use meta.json if present; load: require meta.json; none: do not pass fov_x")
    parser.add_argument("--apply_mask", type=int, choices=[0, 1], default=1,
                        help="Apply model predicted mask to depth output (1=True, 0=False). Default 1.")

    args = parser.parse_args()

    # Runtime configuration.
    input_roots = [args.input] if args.input else INPUT_ROOTS
    output_root = args.output if args.output else OUTPUT_ROOT
    model_path = args.model
    
    # Resolve resize setting
    real_resize = args.resize
    if real_resize <= 0:
        real_resize = None  
    # ===================================================

    # Parameter configuration.
    params = PARAMS.copy()
    params['sampling_ratio'] = args.ratio
    params['resize'] = real_resize 
    params['batch_size'] = args.batch_size # Forward to core pipeline
    params['intrinsics_mode'] = args.intrinsics_mode
    params['apply_mask'] = bool(args.apply_mask)

    print(f"Starting base model inference (Batch Size: {args.batch_size}, Intrinsics: {args.intrinsics_mode})")
    print(f"   Input:  {input_roots}")
    print(f"   Output: {output_root}")
    print(f"   Resize: {real_resize} (CLI value {args.resize})")
    
    run_base_inference_pipeline(
        input_roots=input_roots,
        output_root=output_root,
        model_path=model_path,
        **params
    )
