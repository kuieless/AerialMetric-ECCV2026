import os
import sys
import json
import math  # Used to compute fov_x.
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Dynamic path setup
def setup_moge_path():
    """Find and add the MoGe package path."""
    current_path = Path(__file__).resolve()
    # Walk upward until a directory containing 'moge' is found.
    for parent in [current_path.parents[0], current_path.parents[1], current_path.parents[2]]:
        if (parent / "moge").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    print("Warning: could not locate the moge package path automatically; ensure PYTHONPATH is set.")

setup_moge_path()

try:
    from moge.model import import_model_class_by_version
    import utils3d
except ImportError:
    # Fallback: try to find moge relative to this file
    _repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo_root))
    from moge.model import import_model_class_by_version
    import utils3d

# Base engine

class MogeBaseEngine:
    def __init__(self, model_path, version='v2', device="cuda", fp16=True, intrinsics_mode="none", apply_mask=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        if intrinsics_mode not in {"auto", "load", "none"}:
            raise ValueError("intrinsics_mode must be one of: auto, load, none")
        self.intrinsics_mode = intrinsics_mode
        self.apply_mask = apply_mask
        
        print(f"\n[Base Engine] initializing...")
        print(f"   Model Path: {model_path}")
        print(f"   Version:    {version}")
        print(f"   Intrinsics mode={self.intrinsics_mode}")

        # Load model class.
        MoGeModel = import_model_class_by_version(version)
        
        # Load weights.
        # Base models are loaded via from_pretrained.
        self.model = MoGeModel.from_pretrained(model_path).to(self.device).eval()
        
        if self.fp16:
            self.model.half()
            print("FP16 mode enabled.")
            
        print("Model loaded.")

    @staticmethod
    def _sample_id_from_path(img_path):
        image_dir_names = {"image", "images", "rgb", "rgbs", "color", "jpg"}
        return img_path.stem if img_path.parent.name.lower() in image_dir_names else img_path.parent.name

    def run_scene(self, scene_name, image_paths, output_path, stride=1, resize=1024, batch_size=4):
        output_path = Path(output_path)
        
        if not image_paths: return 0
        if stride > 1: image_paths = image_paths[::stride]
        
        success_count = 0
        with torch.inference_mode():
            # Batched inference loop.
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Scene: {scene_name}", leave=False):
                batch_paths = image_paths[i:i+batch_size]
                try:
                    count = self._process_batch(batch_paths, output_path, resize)
                    success_count += count
                except Exception as e:
                    print(f"Error processing batch near {batch_paths[0].name}: {e}")
        return success_count

    def _process_batch(self, batch_paths, root_out, resize_to):
        images_info = []
        max_h, max_w = 0, 0
        
        # Read images, resize, and load optional intrinsics.
        for img_path in batch_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            h_orig, w_orig = img_bgr.shape[:2]
            
            if resize_to is not None and resize_to > 0:
                scale = resize_to / max(h_orig, w_orig)
                # Only downscale; never upscale.
                if scale < 1.0:
                    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                else:
                    new_w, new_h = w_orig, h_orig
            else:
                new_w, new_h = w_orig, h_orig
                
            # Align dimensions to multiples of 14 for ViT.
            new_w = max(14, (new_w // 14) * 14)
            new_h = max(14, (new_h // 14) * 14)
            
            if (new_w, new_h) != (w_orig, h_orig):
                process_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                process_bgr = img_bgr
                
            img_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1) # [3, H, W]
            
            max_h = max(max_h, new_h)
            max_w = max(max_w, new_w)
            
            fov_x_deg = None 
            
            if self.intrinsics_mode != "none":
                meta_path = img_path.parent / "meta.json"
                if not meta_path.exists():
                    if self.intrinsics_mode == "load":
                        raise FileNotFoundError(f"Missing required intrinsics file: {meta_path}")
                else:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if "intrinsics" not in meta:
                        if self.intrinsics_mode == "load":
                            raise KeyError(f"Missing 'intrinsics' in required file: {meta_path}")
                    else:
                        intr_norm = meta["intrinsics"]
                        # Extract normalized fx from the intrinsics matrix.
                        fx_norm = intr_norm[0][0]
                        # Formula: fov_x = 2 * arctan(0.5 / fx_norm).
                        fov_x_rad = 2 * math.atan(0.5 / fx_norm)
                        fov_x_deg = math.degrees(fov_x_rad)
            
            images_info.append({
                "path": img_path, "orig_shape": (h_orig, w_orig),
                "new_shape": (new_h, new_w), "tensor": tensor,
                "fov_x": fov_x_deg  # Store fov_x in degrees.
            })
            
        if not images_info: return 0
        
        actual_batch_size = len(images_info)
        
        # Pad to the largest image size in the batch.
        batched_tensor = torch.zeros((actual_batch_size, 3, max_h, max_w), device=self.device)
        batched_fov_x = torch.zeros((actual_batch_size,), device=self.device, dtype=torch.float32)
        has_fov = False
        
        if self.fp16: batched_tensor = batched_tensor.half()
        
        for i, info in enumerate(images_info):
            h, w = info["new_shape"]
            batched_tensor[i, :, :h, :w] = info["tensor"].to(self.device)
            # Record fov_x.
            if info["fov_x"] is not None:
                batched_fov_x[i] = info["fov_x"]
                has_fov = True
            
        # Batched inference.
        # Pass fov_x only when intrinsics are available.
        infer_kwargs = {}
        if has_fov:
            infer_kwargs['fov_x'] = batched_fov_x
            
        output = self.model.infer(batched_tensor, **infer_kwargs, apply_mask=self.apply_mask)
            
        depths = output['depth'].cpu().numpy()
        if depths.ndim == 4: depths = depths.squeeze(1) # [B, 1, H, W] -> [B, H, W]
        
        intrinsics_batch = output.get('intrinsics', None)
        if intrinsics_batch is not None: intrinsics_batch = intrinsics_batch.cpu().numpy()

        # Crop padding and resize back to the original resolution.
        for i, info in enumerate(images_info):
            h_orig, w_orig = info["orig_shape"]
            h_new, w_new = info["new_shape"]
            
            # Remove padded regions.
            depth = depths[i, :h_new, :w_new]
            
            # Resize back to the original image size.
            if depth.shape != (h_orig, w_orig):
                depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
            sample_id = self._sample_id_from_path(info["path"])
            save_dir = root_out / sample_id
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(save_dir / 'depth.npy'), depth)
            
            if intrinsics_batch is not None:
                intrinsics = intrinsics_batch[i]
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
                with open(save_dir / 'fov.json', 'w') as f:
                    json.dump({'fov_x': float(np.rad2deg(fov_x)), 'fov_y': float(np.rad2deg(fov_y))}, f)
                    
        return actual_batch_size

class DatasetAutoParser:
    def __init__(self):
        self.scenes = {}
        self.image_dir_names = ("image", "images", "rgb", "rgbs", "color", "jpg")

    def _collect_scene_images(self, scene_dir):
        image_paths = []

        # Norm-style datasets: Scene/SampleID/image.jpg.
        image_paths.extend(scene_dir.rglob("image.jpg"))
        image_paths.extend(scene_dir.rglob("image.png"))

        # Oblique/Wild/raw exports: Scene/rgbs/*.jpg or Scene/image/*.jpg.
        for image_dir_name in self.image_dir_names:
            image_dir = scene_dir / image_dir_name
            if image_dir.is_dir():
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                    image_paths.extend(image_dir.glob(pattern))

        return sorted(set(image_paths))

    def scan(self, root_path):
        root_path = Path(root_path)
        if not root_path.exists(): return
        print(f"Scanning Dataset: {root_path} ...")
        
        for scene_dir in root_path.iterdir():
            if not scene_dir.is_dir(): continue
            
            image_paths = self._collect_scene_images(scene_dir)
            if image_paths:
                self.scenes[scene_dir.name] = {
                    'source_root': root_path,
                    'image_paths': image_paths
                }

# Public pipeline API

def run_base_inference_pipeline(
    input_roots, 
    output_root, 
    model_path, 
    version="v2",
    sampling_ratio=1.0, 
    resize=1024,
    device="cuda",
    batch_size=4,
    intrinsics_mode="none",
    apply_mask=True,
):
    """
    Base model inference entry point.
    """
    # Scan datasets.
    parser = DatasetAutoParser()
    if isinstance(input_roots, str): input_roots = [input_roots]
    for root in input_roots:
        parser.scan(root)
    
    tasks = parser.scenes
    print(f"Found {len(tasks)} scenes to process.")
    if not tasks: return

    # Initialize the engine once.
    engine = MogeBaseEngine(model_path, version=version, device=device, intrinsics_mode=intrinsics_mode, apply_mask=apply_mask)
    
    # Run inference.
    stride = int(1 / sampling_ratio) if sampling_ratio < 1.0 else 1
    
    for i, (scene_name, task) in enumerate(tasks.items()):
        output_dir = Path(output_root) / scene_name
        
        print(f"[{i+1}/{len(tasks)}] {scene_name} (images: {len(task['image_paths'])})")
        
        count = engine.run_scene(
            scene_name=scene_name,
            image_paths=task['image_paths'],
            output_path=output_dir,
            stride=stride,
            resize=resize,
            batch_size=batch_size,
        )
        print(f"   Completed {count} images")
