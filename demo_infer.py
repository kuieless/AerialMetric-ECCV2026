#!/usr/bin/env python3
"""
MoGe2 Depth Inference Demo — single image / video / folder
Output per frame:
  *_depth.npy          — raw float32 depth array
  *_depth_vis.png      — jet colormap visualization
  *_colorbar.png       — standalone colorbar with scale
  *_depth_annotated.png — depth_vis + colorbar stacked, with 2 random point markers
  *_meta.json          — per-frame metadata (vmin/vmax, point depths, shape)

Usage:
  python demo_infer.py --input photo.jpg --model vitl --checkpoint /path/to/vitl-normal.pt
  python demo_infer.py --input video.mp4 --model lora --checkpoint /path/to/Moge2-Aerial.pt --lora_config /path/to/config-lora-all.json
"""

import argparse, json, math, os, random, sys, time
from pathlib import Path

import cv2, numpy as np, torch

# ── path setup ──────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_moge_root = _script_dir / "MoGe"
if _moge_root.exists():
    sys.path.insert(0, str(_moge_root))

from moge.model import import_model_class_by_version


# ══════════════════════════════════════════════════════════════════════
#  Colormap & Drawing Helpers
# ══════════════════════════════════════════════════════════════════════

def apply_colormap(depth, vmin=None, vmax=None, cmap=cv2.COLORMAP_JET):
    """Normalize depth to 0-255 and apply OpenCV colormap. Returns BGR uint8."""
    if vmin is None:
        vmin = float(np.percentile(depth, 2))
    if vmax is None:
        vmax = float(np.percentile(depth, 98))
    vis = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    vis = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cmap)


def save_colorbar(out_path, vmin, vmax, dpi=100):
    """Save a standalone horizontal colorbar PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(6, 0.4))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="jet"),
        cax=ax,
        orientation="horizontal",
        label="Depth (m)",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def colorbar_array(width, height, vmin, vmax, font_scale=0.5):
    """Generate a BGR colorbar strip as a numpy array (height x width x 3)."""
    bar = np.linspace(vmax, vmin, width, dtype=np.float32)
    bar = np.clip((bar - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    bar = (bar * 255).astype(np.uint8)
    bar_bgr = cv2.applyColorMap(bar.reshape(1, -1), cv2.COLORMAP_JET)
    bar_bgr = cv2.resize(bar_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    # Draw tick labels
    bar_bgr = bar_bgr.copy()
    num_ticks = 5
    for i in range(num_ticks):
        val = vmin + (vmax - vmin) * i / (num_ticks - 1)
        x = int(width * i / (num_ticks - 1))
        x = max(0, min(width - 1, x))
        cv2.line(bar_bgr, (x, 0), (x, 5), (255, 255, 255), 1)
        cv2.putText(bar_bgr, f"{val:.1f}", (x + 3, height - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return bar_bgr


def pick_two_valid_points(depth, mask=None, margin=20, seed=None):
    """Pick two random valid-depth points far apart. Returns [(x1,y1), (x2,y2)]."""
    if seed is not None:
        random.seed(seed)
    h, w = depth.shape
    valid = (depth > 0) & np.isfinite(depth)
    if mask is not None:
        valid = valid & (mask > 0)
    ys, xs = np.where(valid)
    if len(xs) < 2:
        return [(w // 3, h // 3), (2 * w // 3, 2 * h // 3)]

    # Filter to interior (avoid edges)
    interior = (xs >= margin) & (xs < w - margin) & (ys >= margin) & (ys < h - margin)
    if interior.sum() >= 2:
        xs, ys = xs[interior], ys[interior]

    # Pick two points: try multiple times to get far-apart pair
    best_dist = -1
    best_pair = None
    indices = list(range(len(xs)))
    for _ in range(min(50, len(xs) // 2)):
        i1, i2 = random.sample(indices, 2)
        dist = (xs[i1] - xs[i2]) ** 2 + (ys[i1] - ys[i2]) ** 2
        if dist > best_dist:
            best_dist = dist
            best_pair = ((int(xs[i1]), int(ys[i1])), (int(xs[i2]), int(ys[i2])))
    return best_pair


def draw_point_marker(img, x, y, depth_val, color=(0, 255, 0), radius=8, font_scale=0.7):
    """Draw a crosshair + depth label on a BGR image."""
    # Crosshair
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, radius * 2, 2, cv2.LINE_AA)
    # Circle
    cv2.circle(img, (x, y), radius, color, 2, cv2.LINE_AA)
    # Label with background
    label = f"P: {depth_val:.2f}m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tx, ty = x + radius + 6, y - th // 2
    if tx + tw > img.shape[1] - 4:
        tx = x - tw - radius - 6
    if ty < 0:
        ty = 4
    if ty + th > img.shape[0] - 4:
        ty = img.shape[0] - th - 4
    cv2.rectangle(img, (tx - 3, ty - 3), (tx + tw + 3, ty + th + 3), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty + th - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_base_model(checkpoint_path, device="cuda", fp16=True):
    """Load MoGe-2 base (non-LoRA) checkpoint."""
    print(f"[Base Model] loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    version = cfg.get("model_version", "v2")
    ModelCls = import_model_class_by_version(version)
    model = ModelCls(**cfg.get("model", {}))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    if fp16:
        model = model.half()
    print(f"  version={version}, fp16={fp16}, device={device}")
    return model


def load_lora_model(config_path, lora_path, lora_rank=96, device="cuda", fp16=True):
    """Load MoGe-2 + LoRA checkpoint with robust key mapping."""
    from peft import LoraConfig, get_peft_model

    print(f"[LoRA Model] loading config={config_path}  weights={lora_path}")
    with open(config_path) as f:
        train_cfg = json.load(f)
    version = train_cfg.get("model_version", "v2")
    ModelCls = import_model_class_by_version(version)
    model = ModelCls(**train_cfg["model"])

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        bias="none",
        target_modules=["qkv", "proj", "fc1", "fc2"],
        modules_to_save=["scale_head"],
    )
    model = get_peft_model(model, lora_cfg)
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    # ── robust key remapping ──
    new_state = {}
    model_keys = set(model.state_dict().keys())
    for k, v in state.items():
        if k in model_keys:
            new_state[k] = v
            continue
        # Try prefix variants
        for prefix in ("base_model.model.", "model.", ""):
            pk = f"{prefix}{k}" if prefix else k
            if pk in model_keys:
                new_state[pk] = v
                break
            # base_layer remap
            parts = pk.split(".")
            if parts[-1] in ("weight", "bias"):
                bik = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if bik in model_keys:
                    new_state[bik] = v
                    break
            # original_module remap (newer PEFT)
            if len(parts) >= 2:
                oik = ".".join(parts[:-1] + ["original_module", parts[-1]])
                if oik in model_keys:
                    new_state[oik] = v
                    break
        else:
            # scale_head / modules_to_save
            for head_name in ("scale_head",):
                if k.startswith(head_name):
                    tk = f"base_model.model.{head_name}.modules_to_save.default.{k[len(head_name)+1:]}"
                    if tk in model_keys:
                        new_state[tk] = v
                        break

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys, e.g.: {missing[:3]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys, e.g.: {unexpected[:3]}")

    model.to(device).eval()
    if fp16:
        model = model.half()
    print(f"  version={version}, rank={lora_rank}, fp16={fp16}")
    return model


# ══════════════════════════════════════════════════════════════════════
#  Inference Core
# ══════════════════════════════════════════════════════════════════════

def intrinsics_to_fov(intrinsics, width, height):
    """Convert 3×3 intrinsics matrix to horizontal/vertical FoV in degrees.

    MoGe outputs normalized intrinsics (cx≈0.5, cy≈0.5), not pixel units.
    For normalized coords: fov = 2 * arctan(0.5 / focal)
    For pixel coords:    fov = 2 * arctan((width/2) / focal)

    Args:
        intrinsics: np.array [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or torch.Tensor
    Returns:
        dict with fov_x_deg, fov_y_deg, fx, fy, cx, cy, is_normalized
    """
    if intrinsics is None:
        return {"fov_x_deg": None, "fov_y_deg": None, "fx": None, "fy": None, "cx": None, "cy": None,
                "is_normalized": None}
    if isinstance(intrinsics, torch.Tensor):
        intr = intrinsics.detach().cpu().float().numpy()
    else:
        intr = np.asarray(intrinsics, dtype=np.float32)
    if intr.ndim == 3:
        intr = intr[0]  # squeeze batch dim
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    # Detect normalized vs pixel intrinsics: cx≈0.5 → normalized
    is_normalized = abs(cx) < 10.0 and abs(cy) < 10.0

    if is_normalized:
        # Normalized coordinates: image span is [0,1], center at 0.5
        fov_x = 2.0 * math.atan(0.5 / fx) * 180.0 / math.pi if fx > 0 else None
        fov_y = 2.0 * math.atan(0.5 / fy) * 180.0 / math.pi if fy > 0 else None
    else:
        # Pixel coordinates: convert to pixel units
        fov_x = 2.0 * math.atan(0.5 * width / fx) * 180.0 / math.pi if fx > 0 else None
        fov_y = 2.0 * math.atan(0.5 * height / fy) * 180.0 / math.pi if fy > 0 else None

    # Diagonal FoV: 2 * arctan( sqrt(tan²(fov_x/2) + tan²(fov_y/2)) )
    if fov_x and fov_y:
        half_diag = math.sqrt(math.tan(math.radians(fov_x / 2)) ** 2 +
                              math.tan(math.radians(fov_y / 2)) ** 2)
        fov_diag = 2.0 * math.degrees(math.atan(half_diag))
    else:
        fov_diag = None

    return {
        "fov_x_deg": round(fov_x, 3) if fov_x else None,
        "fov_y_deg": round(fov_y, 3) if fov_y else None,
        "fov_diag_deg": round(fov_diag, 3) if fov_diag else None,
        "fx": round(fx, 3),
        "fy": round(fy, 3),
        "cx": round(cx, 5),
        "cy": round(cy, 5),
        "is_normalized": is_normalized,
    }


def infer_single(model, img_bgr, device="cuda", fp16=True, resize=None,
                 resolution_level=9, force_projection=True, apply_mask=True,
                 intrinsics_mode="none", given_fov_x=None):
    """Run inference on a single BGR image.

    Args:
        resolution_level: 0-9, higher = better quality, slower (default 9)
        force_projection: recompute point map from depth+intrinsics for consistency
        apply_mask:     mask out invalid regions in output
        given_fov_x:    optional horizontal FoV override (degrees)

    Returns:
        depth      — (H, W) float32 numpy array in meters
        intrinsics — 3×3 numpy array or None
        elapsed    — inference time in seconds
    """
    h, w = img_bgr.shape[:2]

    # ── resize to multiple-of-14 ──
    if resize and resize > 0:
        scale = resize / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = w, h
    new_w = max(14, (new_w // 14) * 14)
    new_h = max(14, (new_h // 14) * 14)

    if (new_w, new_h) != (w, h):
        process = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        process = img_bgr

    # ── BGR→RGB, normalize, to tensor ──
    tensor = (
        torch.from_numpy(cv2.cvtColor(process, cv2.COLOR_BGR2RGB))
        .float()
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    if fp16:
        tensor = tensor.half()

    # ── Build infer() kwargs ──
    infer_kwargs = {
        "resolution_level": resolution_level,
        "force_projection": force_projection,
        "apply_mask": apply_mask,
        "use_fp16": fp16,
    }
    if given_fov_x is not None:
        infer_kwargs["fov_x"] = given_fov_x
    elif intrinsics_mode == "auto":
        infer_kwargs["fov_x"] = None  # let model estimate

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.infer(tensor, **infer_kwargs)
    elapsed = time.perf_counter() - t0

    depth = out["depth"].cpu().float().squeeze().numpy()
    depth = depth[:new_h, :new_w]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Extract intrinsics (model always estimates them) ──
    intr_raw = out.get("intrinsics", None)
    if intr_raw is not None:
        intr_raw = intr_raw.cpu().float().squeeze().numpy()
    return depth, intr_raw, elapsed


# ══════════════════════════════════════════════════════════════════════
#  Output Compositing
# ══════════════════════════════════════════════════════════════════════

def compose_annotated_output(depth, vmin=None, vmax=None, margin=40, seed=None):
    """Produce a single annotated image: depth_vis on top, colorbar on bottom,
    with two random points marked on the depth_vis and their depth values labeled.

    Returns:
        annotated_bgr  — BGR uint8 image
        point_info     — list of dicts with x, y, depth
    """
    if vmin is None:
        vmin = float(np.percentile(depth, 2))
    if vmax is None:
        vmax = float(np.percentile(depth, 98))

    h, w = depth.shape
    bar_height = 40
    gap = 8

    # ── depth visualization ──
    vis = apply_colormap(depth, vmin=vmin, vmax=vmax)

    # ── pick two random points ──
    pts = pick_two_valid_points(depth, margin=margin, seed=seed)
    point_info = []
    for (px, py) in pts:
        d = float(depth[py, px])
        point_info.append({"x": px, "y": py, "depth_m": d})

    # Draw points on a copy so originals stay clean
    vis_annotated = vis.copy()
    colors = [(0, 255, 0), (0, 220, 255)]  # green, yellow-cyan
    for (px, py), color, pi in zip(pts, colors, point_info):
        draw_point_marker(vis_annotated, px, py, pi["depth_m"], color=color)

    # ── colorbar strip ──
    bar = colorbar_array(w, bar_height, vmin, vmax)

    # ── stack: vis_annotated + gap + colorbar ──
    gap_strip = np.full((gap, w, 3), 255, dtype=np.uint8)
    composite = np.vstack([vis_annotated, gap_strip, bar])

    return composite, point_info


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="MoGe2 Depth Inference Demo — enhanced version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_infer.py --input photo.jpg --model vitl --checkpoint model.pt
  python demo_infer.py --input video.mp4 --model lora --checkpoint aerial.pt --lora_config cfg.json --resize 1024
        """,
    )
    # ── Required ──
    p.add_argument("--input", required=True, help="Image / video / folder path")
    p.add_argument("--model", choices=["vitl", "lora"], required=True, help="Model type")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")

    # ── LoRA ──
    p.add_argument("--lora_config", default=None, help="LoRA config JSON (required for --model lora)")
    p.add_argument("--lora_rank", type=int, default=96, help="LoRA rank (default: 96)")

    # ── Output control ──
    p.add_argument("--output", default="./demo_output", help="Output directory (default: ./demo_output)")
    p.add_argument("--save_npy", action="store_true", default=True, help="Save raw .npy depth file")
    p.add_argument("--no_npy", action="store_false", dest="save_npy", help="Skip saving raw .npy")
    p.add_argument("--save_components", action="store_true", default=False,
                   help="Also save individual vis/colorbar files (not just composite)")

    # ── Inference parameters ──
    p.add_argument("--resize", type=int, default=0,
                   help="Long-edge resize target (0=original size). Padded to multiple of 14.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: cuda / cpu")
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 (default: True on CUDA)")
    p.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable FP16")
    p.add_argument("--intrinsics_mode", choices=["auto", "load", "none"], default="none",
                   help="Intrinsics handling mode (default: none)")
    p.add_argument("--fov_x", type=float, default=None,
                   help="Override horizontal FoV in degrees (default: auto-estimate by model)")

    # ── MoGe2 model hyperparameters ──
    p.add_argument("--resolution_level", type=int, default=9, choices=range(0, 10),
                   help="MoGe2 quality/speed tradeoff: 0=fastest, 9=best (default: 9)")
    p.add_argument("--force_projection", action="store_true", default=True,
                   help="Recompute point map from depth for consistency (default: on)")
    p.add_argument("--no_force_projection", action="store_false", dest="force_projection",
                   help="Skip projection recompute (faster, may be less consistent)")
    p.add_argument("--apply_mask", action="store_true", default=True,
                   help="Mask invalid depth regions (default: on)")
    p.add_argument("--no_apply_mask", action="store_false", dest="apply_mask",
                   help="Keep all depth values including invalid")

    # ── Video / folder ──
    p.add_argument("--stride", type=int, default=1, help="Frame stride for video (default: 1)")

    # ── Visualization ──
    p.add_argument("--cmap", default="jet", choices=["jet", "inferno", "plasma", "viridis", "turbo"],
                   help="Colormap (default: jet)")
    p.add_argument("--vmin", type=float, default=None, help="Depth min for colormap (auto by default)")
    p.add_argument("--vmax", type=float, default=None, help="Depth max for colormap (auto by default)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for point picking (default: 42)")
    p.add_argument("--point_margin", type=int, default=20,
                   help="Edge margin when picking random points (default: 20)")

    args = p.parse_args()

    # ── Validate ──
    if args.model == "lora" and not args.lora_config:
        p.error("--lora_config is required for --model lora")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {args.input}")

    os.makedirs(args.output, exist_ok=True)
    device = args.device
    resize = args.resize if args.resize > 0 else None
    fp16 = args.fp16 and device.startswith("cuda")

    # Map cmap name → OpenCV constant
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    cmap = cmap_map.get(args.cmap, cv2.COLORMAP_JET)

    # ── Load model ──
    t0 = time.perf_counter()
    if args.model == "lora":
        model = load_lora_model(args.lora_config, args.checkpoint, lora_rank=args.lora_rank,
                                device=device, fp16=fp16)
    else:
        model = load_base_model(args.checkpoint, device=device, fp16=fp16)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # ── Collect frames ──
    frames = []  # list of (stem_name, bgr_image)

    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        imgs = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        for p in imgs:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append((p.stem, img))
            else:
                print(f"[WARN] Cannot read: {p}")
        print(f"Found {len(frames)} images in {input_path}")
    elif input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {input_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        pbar = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % args.stride == 0:
                frames.append((f"frame_{idx:06d}", frame))
            idx += 1
        cap.release()
        print(f"Extracted {len(frames)} frames from video (stride={args.stride}, total={total_frames})")
    else:
        img = cv2.imread(str(input_path))
        if img is None:
            raise SystemExit(f"Cannot read image: {input_path}")
        frames.append((input_path.stem, img))

    if not frames:
        raise SystemExit("No frames to process — check input path and supported formats")

    # ── Run inference ──
    total_time = 0.0
    for i, (name, img) in enumerate(frames):
        h, w = img.shape[:2]
        print(f"\n[{i+1}/{len(frames)}] {name}  ({w}x{h})")

        depth, intr, elapsed = infer_single(
            model, img, device=device, fp16=fp16,
            resize=resize,
            resolution_level=args.resolution_level,
            force_projection=args.force_projection,
            apply_mask=args.apply_mask,
            intrinsics_mode=args.intrinsics_mode,
            given_fov_x=args.fov_x,
        )
        total_time += elapsed

        # ── FoV from intrinsics ──
        fov_info = intrinsics_to_fov(intr, w, h)

        print(f"  Inference: {elapsed:.2f}s  |  depth range: [{depth.min():.2f}, {depth.max():.2f}] m")
        if fov_info["fov_x_deg"]:
            print(f"  FoV: x={fov_info['fov_x_deg']}°  y={fov_info['fov_y_deg']}°  "
                  f"diag={fov_info['fov_diag_deg']}°  "
                  f"fx={fov_info['fx']:.1f} fy={fov_info['fy']:.1f}")

        # ── Compute vmin/vmax ──
        vmin = args.vmin if args.vmin is not None else float(np.percentile(depth, 2))
        vmax = args.vmax if args.vmax is not None else float(np.percentile(depth, 98))

        # ── 1. Raw depth .npy ──
        if args.save_npy:
            npy_path = os.path.join(args.output, f"{name}_depth.npy")
            np.save(npy_path, depth)
            print(f"  Saved: {npy_path}  ({os.path.getsize(npy_path)/1024:.0f} KB)")

        # ── 2. Composite annotated output (vis + points + colorbar) ──
        composite, point_info = compose_annotated_output(
            depth, vmin=vmin, vmax=vmax, margin=args.point_margin, seed=args.seed + i
        )
        annotated_path = os.path.join(args.output, f"{name}_depth_annotated.png")
        cv2.imwrite(annotated_path, composite)
        print(f"  Saved: {annotated_path}  ({os.path.getsize(annotated_path)/1024:.0f} KB)")
        for j, pi in enumerate(point_info):
            print(f"    Point {j+1}: ({pi['x']:4d}, {pi['y']:4d})  depth = {pi['depth_m']:.3f} m")

        # ── 3. Standalone colorbar ──
        colorbar_path = os.path.join(args.output, f"{name}_colorbar.png")
        save_colorbar(colorbar_path, vmin, vmax)

        # ── 4. Standalone depth visualization (no points) ──
        if args.save_components:
            vis = apply_colormap(depth, vmin=vmin, vmax=vmax, cmap=cmap)
            vis_path = os.path.join(args.output, f"{name}_depth_vis.png")
            cv2.imwrite(vis_path, vis)

        # ── 5. Per-frame metadata JSON ──
        meta = {
            "name": name,
            "image_size": [w, h],
            "depth_range": [round(vmin, 3), round(vmax, 3)],
            "intrinsics": fov_info,
            "points": point_info,
            "inference_time_s": round(elapsed, 3),
            "model_config": {
                "model_type": args.model,
                "checkpoint": args.checkpoint,
                "lora_config": args.lora_config,
                "lora_rank": args.lora_rank if args.model == "lora" else None,
                "resolution_level": args.resolution_level,
                "force_projection": args.force_projection,
                "apply_mask": args.apply_mask,
                "intrinsics_mode": args.intrinsics_mode,
                "fov_x_override": args.fov_x,
                "resize": args.resize,
                "fp16": fp16,
            },
        }
        meta_path = os.path.join(args.output, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── Summary ──
    n = len(frames)
    print(f"\n{'='*60}")
    print(f"Done. {n} frame(s) → {args.output}/")
    print(f"  Total inference: {total_time:.1f}s  |  Avg: {total_time/n:.2f}s/frame")
    print(f"  Per frame: *_depth.npy  *_depth_annotated.png  *_colorbar.png  *_meta.json")
    if args.save_components:
        print(f"             *_depth_vis.png  (standalone)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
