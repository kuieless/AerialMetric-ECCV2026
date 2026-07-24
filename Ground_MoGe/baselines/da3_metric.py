# Reference: https://github.com/DepthAnything/Depth-Anything-3
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import click
import torch
import torch.nn.functional as F

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    def __init__(
        self,
        repo_path: str,
        model_name_or_path: str,
        process_res: int,
        process_res_method: str,
        default_focal_px: float,
        da3_log_level: str,
        device: str,
    ):
        repo_path = os.path.abspath(repo_path)
        if Path(repo_path).exists():
            repo_src = os.path.join(repo_path, "src")
            if repo_src not in sys.path:
                sys.path.insert(0, repo_src)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        # Control Depth-Anything-3 internal logger verbosity before import.
        os.environ["DA3_LOG_LEVEL"] = da3_log_level.upper()
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(model_name_or_path).to(self.device).eval()
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.default_focal_px = float(default_focal_px)

    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../Depth-Anything-3", help="Path to Depth-Anything-3 repository.")
    @click.option("--model", "model_name_or_path", type=str, default="depth-anything/DA3METRIC-LARGE", help="HuggingFace model id or local model path.")
    @click.option("--process_res", type=int, default=504, help="DA3 internal process resolution.")
    @click.option(
        "--process_res_method",
        type=click.Choice(["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"]),
        default="upper_bound_resize",
        help="DA3 internal resize method.",
    )
    @click.option("--default_focal_px", type=float, default=1000.0, help="Fallback focal length (pixels) if intrinsics are unavailable.")
    @click.option(
        "--da3_log_level",
        type=click.Choice(["ERROR", "WARN", "INFO", "DEBUG"], case_sensitive=False),
        default="ERROR",
        help="Depth-Anything-3 internal log level. ERROR suppresses per-image INFO logs.",
    )
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    @staticmethod
    def load(
        repo_path: str,
        model_name_or_path: str,
        process_res: int,
        process_res_method: str,
        default_focal_px: float,
        da3_log_level: str,
        device: str,
    ):
        return Baseline(repo_path, model_name_or_path, process_res, process_res_method, default_focal_px, da3_log_level, device)

    def _to_rgb_uint8(self, image: torch.Tensor) -> "np.ndarray":
        import numpy as np

        image_np = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (image_np * 255).astype(np.uint8)

    def _infer_single(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_h, target_w = image.shape[-2:]
        prediction = self.model.inference(
            [self._to_rgb_uint8(image)],
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )
        depth = torch.from_numpy(prediction.depth[0].astype("float32")).to(self.device)
        proc_h, proc_w = depth.shape[-2:]
        # DA3Metric's net output is defined in the processed-image focal space.
        focal_px = self._get_focal_px(intrinsics, proc_w, proc_h)
        depth = depth * (focal_px / 300.0)
        if depth.shape[-2:] != (target_h, target_w):
            depth = F.interpolate(
                depth[None, None],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return depth

    def _get_focal_px(self, intrinsics: Optional[torch.Tensor], width: int, height: int) -> float:
        if intrinsics is None:
            return self.default_focal_px
        intrinsics = intrinsics.detach()
        fx_px = float(intrinsics[0, 0].item() * width)
        fy_px = float(intrinsics[1, 1].item() * height)
        focal_px = 0.5 * (fx_px + fy_px)
        return focal_px if focal_px > 1e-6 else self.default_focal_px

    @torch.inference_mode()
    def infer(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if image.ndim == 3:
            depth = self._infer_single(image, intrinsics)
        else:
            depth = torch.stack(
                [
                    self._infer_single(image[i], intrinsics[i] if intrinsics is not None else None)
                    for i in range(image.shape[0])
                ],
                dim=0,
            )
        return {"depth_metric": depth}
