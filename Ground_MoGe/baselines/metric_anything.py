# Reference: https://github.com/metric-anything/metric-anything
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import click
import torch
import torchvision.transforms.functional as TF

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    def __init__(
        self,
        repo_path: str,
        pretrained_model_name_or_path: str,
        checkpoint_path: str,
        checkpoint_filename: str,
        default_focal_px: float,
        device: str,
    ):
        repo_path = os.path.abspath(repo_path)
        student_depthmap_dir = os.path.join(repo_path, "models", "student_depthmap")
        if not Path(student_depthmap_dir).exists():
            raise FileNotFoundError(
                f"Cannot find metric-anything student_depthmap directory at {student_depthmap_dir}"
            )
        if student_depthmap_dir not in sys.path:
            sys.path.insert(0, student_depthmap_dir)

        from depth_model import MetricAnythingDepthMap

        self.device = torch.device(device)
        self.default_focal_px = float(default_focal_px)

        # metric-anything uses torch.hub.load("network", source="local"), which depends on CWD.
        # Switch CWD temporarily so it always resolves to student_depthmap/network/hubconf.py.
        old_cwd = os.getcwd()
        os.chdir(student_depthmap_dir)
        try:
            ckpt_path = Path(checkpoint_path) if checkpoint_path else None
            if ckpt_path and ckpt_path.exists():
                model = MetricAnythingDepthMap.from_pretrained(
                    str(ckpt_path),
                    model_kwargs={"device": device},
                )
            else:
                model = MetricAnythingDepthMap.from_pretrained(
                    pretrained_model_name_or_path,
                    model_kwargs={"device": device},
                    filename=checkpoint_filename,
                )
        finally:
            os.chdir(old_cwd)

        self.model = model.to(self.device).eval()
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)

    @staticmethod
    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../metric-anything", help="Path to metric-anything repository.")
    @click.option(
        "--pretrained",
        "pretrained_model_name_or_path",
        type=str,
        default="yjh001/metricanything_student_depthmap",
        help="HuggingFace model id or local checkpoint path.",
    )
    @click.option("--checkpoint", "checkpoint_path", type=click.Path(), default="", help="Optional local checkpoint file path.")
    @click.option("--checkpoint_filename", type=str, default="student_depthmap.pt", help="Checkpoint filename on HuggingFace.")
    @click.option("--default_focal_px", type=float, default=1000.0, help="Fallback focal length in pixels when intrinsics are unavailable.")
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    def load(
        repo_path: str = "../metric-anything",
        pretrained_model_name_or_path: str = "yjh001/metricanything_student_depthmap",
        checkpoint_path: str = "",
        checkpoint_filename: str = "student_depthmap.pt",
        default_focal_px: float = 1000.0,
        device: str = "cuda",
    ):
        return Baseline(
            repo_path=repo_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint_path=checkpoint_path,
            checkpoint_filename=checkpoint_filename,
            default_focal_px=default_focal_px,
            device=device,
        )

    def _get_focal_px(self, intrinsics: Optional[torch.Tensor], width: int, height: int) -> float:
        if intrinsics is None:
            return self.default_focal_px if self.default_focal_px > 0 else float(width)
        fx_px = float(intrinsics[0, 0].item() * width)
        fy_px = float(intrinsics[1, 1].item() * height)
        focal = 0.5 * (fx_px + fy_px)
        if focal <= 1e-6:
            return self.default_focal_px if self.default_focal_px > 0 else float(width)
        return focal

    @torch.inference_mode()
    def _infer_single(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = image.shape[-2:]
        focal_px = self._get_focal_px(intrinsics, w, h)
        x = TF.normalize(image.unsqueeze(0), mean=self._mean, std=self._std)
        pred = self.model.infer(x.to(self.device), f_px=focal_px)
        depth = pred["depth"]
        if depth.ndim == 3:
            depth = depth[0]
        return depth.to(self.device).float()

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
