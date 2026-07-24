
import os
import sys
import json
import math  # Used to compute fov_x.
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# Dynamic path setup
def setup_moge_path():
    current_path = Path(__file__).resolve()
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
    _repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo_root))
    from moge.model import import_model_class_by_version
    import utils3d

# Core engine

class MogeLoRAEngine:
    def __init__(self, config_path, lora_path, device="cuda", fp16=True, lora_rank=96, intrinsics_mode="none", apply_mask=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.lora_rank = lora_rank
        self.lora_alpha = 2 * lora_rank
        if intrinsics_mode not in {"auto", "load", "none"}:
            raise ValueError("intrinsics_mode must be one of: auto, load, none")
        self.intrinsics_mode = intrinsics_mode
        self.apply_mask = apply_mask
        
        print(f"\n[LoRA Engine] initializing...")
        print(f"   LoRA rank={self.lora_rank}, alpha={self.lora_alpha}")
        print(f"   Intrinsics mode={self.intrinsics_mode}")
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        
        model_version = train_config.get('model_version', 'v2')
        MoGeModel = import_model_class_by_version(model_version)
        self.model = MoGeModel(**train_config['model'])
        
        # LoRA Config
        LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
        HEADS_TO_SAVE = ["scale_head"] 
        peft_config = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha, bias="none",
            target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE 
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Load Weights
        print("   Loading LoRA weights...")
        checkpoint = torch.load(lora_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        model_keys = set(self.model.state_dict().keys())
        
        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v; continue
            prefixed_k = f"base_model.model.{k}"
            if prefixed_k in model_keys:
                new_state_dict[prefixed_k] = v; continue
            parts = prefixed_k.split('.')
            if parts[-1] in ['weight', 'bias']:
                base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if base_injected_k in model_keys:
                    new_state_dict[base_injected_k] = v; continue
            for head in HEADS_TO_SAVE:
                if k.startswith(head):
                    suffix = k[len(head)+1:]
                    trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                    if trainable_k in model_keys:
                        new_state_dict[trainable_k] = v; break

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if self.fp16: self.model.half()

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
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Scene: {scene_name}", leave=False):
                batch_paths = image_paths[i:i+batch_size]
                try:
                    count = self._process_batch(batch_paths, output_path, resize)
                    success_count += count
                except Exception as e:
                    print(f"\nError processing batch near {batch_paths[0].parent.name}: {e}")
        return success_count

    def _process_batch(self, batch_paths, root_out, resize_to):
        images_info = []
        max_h, max_w = 0, 0
        
########        # Read images and optional intrinsics.  ####  
        for img_path in batch_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            h_orig, w_orig = img_bgr.shape[:2]
            
            if resize_to is not None:
                scale = resize_to / max(h_orig, w_orig)
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            else:
                new_w, new_h = w_orig, h_orig
                
            new_w = max(14, (new_w // 14) * 14)
            new_h = max(14, (new_h // 14) * 14)
            
            if (new_w, new_h) != (w_orig, h_orig):
                process_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                process_bgr = img_bgr
                
            img_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1)
            
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
                        fx_norm = intr_norm[0][0]
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
            
        # Pass fov_x only when intrinsics are available.
        infer_kwargs = {}
        if has_fov:
            infer_kwargs['fov_x'] = batched_fov_x

        if hasattr(self.model, 'infer'):
            output = self.model.infer(batched_tensor, **infer_kwargs, apply_mask=self.apply_mask)
        elif hasattr(self.model.base_model.model, 'infer'):
            output = self.model.base_model.model.infer(batched_tensor, **infer_kwargs, apply_mask=self.apply_mask)
        else: 
            raise AttributeError("Could not find an infer method")
            
        depths = output['depth'].cpu().numpy()
        if depths.ndim == 4: depths = depths.squeeze(1)
        
        out_intrinsics = output.get('intrinsics', None)
        if out_intrinsics is not None: out_intrinsics = out_intrinsics.cpu().numpy()

        # Save results.
        for i, info in enumerate(images_info):
            h_orig, w_orig = info["orig_shape"]
            h_new, w_new = info["new_shape"]
            
            depth = depths[i, :h_new, :w_new]
            if depth.shape != (h_orig, w_orig):
                depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
            sample_id = self._sample_id_from_path(info["path"])
            save_dir = root_out / sample_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(str(save_dir / 'depth.npy'), depth)
            
            if out_intrinsics is not None:
                intrin = out_intrinsics[i]
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrin)
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
        
        # Iterate over scene folders.
        for scene_dir in root_path.iterdir():
            if not scene_dir.is_dir(): continue
            
            image_paths = self._collect_scene_images(scene_dir)
            if image_paths:
                self.scenes[scene_dir.name] = {
                    'source_root': root_path,
                    'image_paths': image_paths
                }

# Public pipeline API

def run_inference_pipeline(
    input_roots, output_root, lora_config, lora_weight,
    sampling_ratio=1.0, resize=1024, device="cuda", batch_size=4,
    lora_rank=96, intrinsics_mode="none", apply_mask=True,
):
    parser = DatasetAutoParser()
    if isinstance(input_roots, str): input_roots = [input_roots]
    for root in input_roots: parser.scan(root)
    
    tasks = parser.scenes
    print(f"Found {len(tasks)} scenes to process.")
    if not tasks: return

    engine = MogeLoRAEngine(
        lora_config,
        lora_weight,
        device=device,
        lora_rank=lora_rank,
        intrinsics_mode=intrinsics_mode,
        apply_mask=apply_mask,
    )
    stride = int(1 / sampling_ratio) if sampling_ratio < 1.0 else 1
    
    for i, (scene_name, task) in enumerate(tasks.items()):
        output_dir = Path(output_root) / scene_name
        print(f"\n[{i+1}/{len(tasks)}] Scene: {scene_name} (images: {len(task['image_paths'])})")
        
        count = engine.run_scene(
            scene_name=scene_name,
            image_paths=task['image_paths'],
            output_path=output_dir,
            stride=stride,
            resize=resize,
            batch_size=batch_size
        )
        print(f"   Completed {count} images")
