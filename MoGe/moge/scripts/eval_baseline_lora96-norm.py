import os
import sys
from pathlib import Path
import json
import math
from typing import *

import click
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# 自动处理模块路径
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import utils3d
from moge.test.dataloader import EvalDataLoaderPipeline
from moge.test.metrics import compute_metrics
from moge.utils.geometry_torch import intrinsics_to_fov
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.tools import key_average, timeit
from moge.model import import_model_class_by_version

# ==============================================================================
# 🔥 核心类：LoRA 基线模型与批量推理逻辑 (内置 Ultimate Oracle)
# ==============================================================================
class LoRABaseline:
    def __init__(self, lora_config_path, lora_weight_path, device="cuda", fp16=True, lora_rank=96):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.lora_rank = lora_rank
        self.lora_alpha = 2 * lora_rank
        print(f"\n📦 Loading LoRA Model...")
        print(f"   LoRA rank={self.lora_rank}, alpha={self.lora_alpha}")
        
        with open(lora_config_path, 'r') as f:
            train_config = json.load(f)

        model_version = train_config.get('model_version', 'v2')
        MoGeModel = import_model_class_by_version(model_version)
        self.model = MoGeModel(**train_config['model'])

        LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
        HEADS_TO_SAVE = ["scale_head"]
        peft_config = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha, bias="none",
            target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE
        )
        self.model = get_peft_model(self.model, peft_config)

        print(f"   Weight: {lora_weight_path}")
        checkpoint = torch.load(lora_weight_path, map_location='cpu')
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
        print(f"✅ LoRA Loaded. Ready for inference.")

        self.model.to(self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

    def infer_batch_for_evaluation(self, images: list, intrinsics: list = None):
        B = len(images)
        orig_shapes = [(img.shape[1], img.shape[2]) for img in images]
        
        # 1. 计算对齐尺寸
        new_shapes = []
        for h, w in orig_shapes:
            new_h, new_w = (h // 14) * 14, (w // 14) * 14
            new_shapes.append((new_h, new_w))
            
        # 2. 获取最大尺寸用于 Padding
        max_h = max(14, max(s[0] for s in new_shapes))
        max_w = max(14, max(s[1] for s in new_shapes))
        
        # 3. 构造张量与先验 FOV
        batched_tensor = torch.zeros((B, 3, max_h, max_w), device=self.device).half()
        batched_fov_x = torch.zeros((B,), device=self.device, dtype=torch.float32)
        has_fov = False
        
        for i, img in enumerate(images):
            h, w = new_shapes[i]
            h_orig, w_orig = orig_shapes[i]
            
            img = img.to(self.device).half().unsqueeze(0)
            if (h, w) != (h_orig, w_orig):
                img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
            batched_tensor[i, :, :h, :w] = img.squeeze(0)
            
            # 🔥 转换物理像素焦距为角度制 fov_x 供网络做先验
            if intrinsics is not None and intrinsics[i] is not None:
                intr = intrinsics[i]
                fx = intr[0, 0].item()
                fov_x_rad = 2 * math.atan(w_orig / (2 * fx))
                batched_fov_x[i] = math.degrees(fov_x_rad)
                has_fov = True

        infer_kwargs = {}
        if has_fov:
            infer_kwargs['fov_x'] = batched_fov_x

        # 4. 执行推理
        with torch.inference_mode():
            if hasattr(self.model, 'infer'):
                output = self.model.infer(batched_tensor, **infer_kwargs)
            elif hasattr(self.model.base_model.model, 'infer'):
                output = self.model.base_model.model.infer(batched_tensor, **infer_kwargs)
            else:
                output = self.model(batched_tensor, **infer_kwargs)

        # 5. 后处理与逆透视重构点云
        preds = []
        for i in range(B):
            h_orig, w_orig = orig_shapes[i]
            new_h, new_w = new_shapes[i]
            needs_resize = (new_h != h_orig) or (new_w != w_orig)
            
            pred_i = {}
            for k, v in output.items():
                if not isinstance(v, torch.Tensor):
                    pred_i[k] = v
                    continue
                
                v_i = v[i]
                while v_i.dim() > 1 and v_i.shape[0] == 1:
                    v_i = v_i.squeeze(0)
                
                # Case A: Points
                if v_i.ndim == 3 and v_i.shape[0] == max_h and v_i.shape[1] == max_w and v_i.shape[2] == 3:
                    v_i = v_i[:new_h, :new_w, :] 
                    if needs_resize:
                        v_perm = v_i.permute(2, 0, 1).unsqueeze(0)
                        v_res = F.interpolate(v_perm, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0).permute(1, 2, 0)
                
                # Case B: Normal
                elif v_i.ndim == 3 and v_i.shape[0] == 3 and v_i.shape[1] == max_h and v_i.shape[2] == max_w:
                    v_i = v_i[:, :new_h, :new_w]
                    if needs_resize:
                        v_uns = v_i.unsqueeze(0)
                        v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0)
                
                # Case C: Depth/Mask
                elif v_i.ndim == 2 and v_i.shape[0] == max_h and v_i.shape[1] == max_w:
                    v_i = v_i[:new_h, :new_w]
                    if needs_resize:
                        is_bool = (v_i.dtype == torch.bool)
                        if is_bool: v_i = v_i.float()
                        v_uns = v_i.unsqueeze(0).unsqueeze(0)
                        v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0).squeeze(0)
                        if is_bool or k == 'mask':
                            v_i = (v_i > 0.5)

                pred_i[k] = v_i

            # 🔥 Oracle 模式：绝对精度重构
            if intrinsics is not None and intrinsics[i] is not None and 'depth' in pred_i:
                intr = intrinsics[i].to(self.device).float()
                depth = pred_i['depth'].float()
                
                v, u = torch.meshgrid(
                    torch.arange(h_orig, device=self.device, dtype=torch.float32), 
                    torch.arange(w_orig, device=self.device, dtype=torch.float32), 
                    indexing='ij'
                )
                
                fx, fy = intr[0, 0], intr[1, 1]
                cx, cy = intr[0, 2], intr[1, 2]
                
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                
                pred_i['points'] = torch.stack([x, y, z], dim=-1).half() 
                pred_i['intrinsics'] = intr

            if 'depth' in pred_i: pred_i['depth_metric'] = pred_i['depth']
            if 'points' in pred_i: pred_i['points_metric'] = pred_i['points']
            preds.append(pred_i)

        return preds

# ==============================================================================
# 🚀 主程序：评测循环 (直接读取 LoRA 配置，不走 --baseline)
# ==============================================================================
@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Standalone LoRA Evaluation script.')
@click.option('--lora_config', type=click.Path(), required=True, help='Path to the LoRA config.json.')
@click.option('--lora_weight', type=click.Path(), required=True, help='Path to the LoRA .pt file.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations.')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation.')
@click.option('--dump_pred', is_flag=True, help='Dump prediction results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.option('--ratio', type=float, default=1.0, help='Sampling ratio (0.0-1.0). Default 1.0 (100%).')
@click.option('--batch_size', type=int, required=True, help='Batch size for faster evaluation.')
@click.option('--lora_rank', type=click.Choice(['64', '96', '128']), required=True, help='LoRA rank (r); alpha is set to 2 * rank.')
@click.pass_context
def main(ctx: click.Context, lora_config: str, lora_weight: str, config_path: str, oracle_mode: bool, output_path: Union[str, Path], dump_pred: bool, dump_gt: bool, ratio: float, batch_size: int, lora_rank: str):
    
    # 🌟 直接初始化我们封装好的 LoRABaseline
    baseline = LoRABaseline(
        lora_config,
        lora_weight,
        device="cuda",
        lora_rank=int(lora_rank),
    )

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            total_len = len(eval_data_pipe)
            target_limit = int(total_len * ratio)
            
            batch_samples = []
            
            for i in range(total_len):
                if i >= target_limit:
                    break
                    
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                batch_samples.append(sample)
                
                if len(batch_samples) == batch_size or i == min(total_len, target_limit) - 1:
                    images = [s['image'] for s in batch_samples]
                    gt_intrinsics = [s['intrinsics'] for s in batch_samples] if oracle_mode else None

                    torch.cuda.synchronize()
                    with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                        # 核心调用
                        preds = baseline.infer_batch_for_evaluation(images, gt_intrinsics)
                    torch.cuda.synchronize()

                    time_per_sample = timer.time / len(batch_samples)

                    for pred, samp in zip(preds, batch_samples):
                        metrics, misc = compute_metrics(pred, samp, vis=dump_pred or dump_gt)
                        metrics['inference_time'] = time_per_sample
                        metrics_list.append(metrics)

                        dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', samp['filename'].replace('.zip', ''))
                        if dump_pred:
                            dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                            image_cpu = samp['image']
                            cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image_cpu.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                            with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                                json.dump(metrics, f, indent=4)

                            if 'pred_points' in misc:
                                points = misc['pred_points'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            
                            if 'pred_depth' in misc:
                                depth = misc['pred_depth'].cpu().numpy()
                                if 'mask' in pred:
                                    mask = pred['mask'].cpu().numpy()
                                    depth = np.where(mask, depth, np.inf)
                                cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))

                            if 'mask' in pred:
                                mask = pred['mask'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))

                            if 'normal' in pred:
                                normal = pred['normal'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                            if 'intrinsics' in pred:
                                intrinsics = pred['intrinsics']
                                fov_x, fov_y = intrinsics_to_fov(intrinsics)
                                with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                                    json.dump({
                                        'fov_x': np.rad2deg(fov_x.item()),
                                        'fov_y': np.rad2deg(fov_y.item()),
                                        'intrinsics': intrinsics.cpu().numpy().tolist(),
                                    }, f)
                        
                        if dump_gt:
                            dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                            image_cpu = samp['image']
                            cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image_cpu.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                            if 'points' in samp:
                                points = samp['points']
                                cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                            if 'depth' in samp:
                                depth = samp['depth']
                                mask = samp['depth_mask']
                                cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                            if 'normal' in samp:
                                normal = samp['normal']
                                cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                            if 'depth_mask' in samp:
                                mask = samp['depth_mask']
                                cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                            if 'intrinsics' in samp:
                                intrinsics = samp['intrinsics']
                                fov_x, fov_y = intrinsics_to_fov(intrinsics)
                                with open(dump_path / 'gt' / 'info.json', 'w') as f:
                                    json.dump({
                                        'fov_x': np.rad2deg(fov_x.item()),
                                        'fov_y': np.rad2deg(fov_y.item()),
                                        'intrinsics': intrinsics.cpu().numpy().tolist(),
                                    }, f)

                    pbar.update(len(batch_samples))
                    
                    if i % min(100, batch_size * 5) < batch_size or i == min(total_len, target_limit) - 1:
                        Path(output_path).write_text(
                            json.dumps({
                                **all_metrics, 
                                benchmark_name: key_average(metrics_list)
                            }, indent=4)
                        )
                    
                    batch_samples = []

            all_metrics[benchmark_name] = key_average(metrics_list)

    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))

if __name__ == '__main__':
    main()
