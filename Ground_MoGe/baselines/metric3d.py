# Reference: https://github.com/YvanYin/Metric3D
import os
import pickle
from typing import Dict, Literal, Optional

import click
import cv2
import torch
import torch.nn.functional as F

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    def __init__(
        self,
        repo_path: str,
        backbone: Literal["vits", "vitl", "vitg"],
        checkpoint_path: Optional[str],
        default_focal_px: float,
        device: str,
    ):
        backbone_map = {
            "vits": "metric3d_vit_small",
            "vitl": "metric3d_vit_large",
            "vitg": "metric3d_vit_giant2",
        }

        self.device = torch.device(device)
        model_name = backbone_map[backbone]
        repo_path = os.path.abspath(repo_path)
        checkpoint_path = os.path.abspath(checkpoint_path) if checkpoint_path else ""

        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Metric3D checkpoint does not exist: {checkpoint_path}")

        if os.path.exists(repo_path):
            model = torch.hub.load(repo_path, model_name, source="local", pretrain=not bool(checkpoint_path))
        else:
            model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=not bool(checkpoint_path))

        if checkpoint_path:
            checkpoint = self._load_checkpoint(checkpoint_path)
            state_dict = self._extract_state_dict(checkpoint)
            state_dict = self._strip_common_prefix(state_dict, "module.")
            state_dict = self._strip_common_prefix(state_dict, "model.")
            model.load_state_dict(state_dict, strict=False)

        self.model = model.to(self.device).eval()
        self.default_focal_px = float(default_focal_px)

    @staticmethod
    def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format: expected dict, got {type(checkpoint)!r}")

        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    @staticmethod
    def _strip_common_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
        return state_dict

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict:
        try:
            return torch.load(checkpoint_path, map_location="cpu")
        except pickle.UnpicklingError as exc:
            # PyTorch 2.6 changed torch.load default to weights_only=True.
            # Metric3D checkpoints may include config objects (for example ConfigDict),
            # so trusted local checkpoints need a fallback with weights_only=False.
            if "Weights only load failed" not in str(exc):
                raise
            try:
                return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(checkpoint_path, map_location="cpu")

    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../Metric3D", help="Path to Metric3D repository. Falls back to torch.hub remote if missing.")
    @click.option("--backbone", type=click.Choice(["vits", "vitl", "vitg"]), default="vitg", help="Backbone architecture.")
    @click.option("--checkpoint", "checkpoint_path", type=click.Path(), default="", help="Optional local Metric3D checkpoint (.pth). If set, load this checkpoint instead of the default pretrained weights.")
    @click.option("--default_focal_px", type=float, default=1000.0, help="Fallback focal length in pixels when intrinsics are unavailable.")
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    @staticmethod
    def load(
        repo_path: str = "../Metric3D",
        backbone: str = "vitg",
        checkpoint_path: str = "",
        default_focal_px: float = 1000.0,
        device: str = "cuda",
    ):
        return Baseline(repo_path, backbone, checkpoint_path, default_focal_px, device)

    def _prepare_input(self, image: torch.Tensor):
        rgb_origin = image.detach().cpu().numpy().transpose((1, 2, 0)) * 255.0
        input_size = (616, 1064)  # (H, W) for ViT models
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        resized_w, resized_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb_origin, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        padding = [123.675, 116.28, 103.53]
        pad_h = input_size[0] - resized_h
        pad_w = input_size[1] - resized_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = (pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half)

        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=self.device)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=self.device)[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float().to(self.device)
        rgb = ((rgb - mean) / std)[None, ...]

        return rgb, pad_info, resized_w, resized_h

    def _intrinsics_to_focal_px(self, intrinsics: Optional[torch.Tensor], width: int, height: int) -> float:
        if intrinsics is None:
            return self.default_focal_px
        fx_px = float(intrinsics[0, 0].item() * width)
        fy_px = float(intrinsics[1, 1].item() * height)
        focal = 0.5 * (fx_px + fy_px)
        return focal if focal > 1e-6 else self.default_focal_px

    @torch.inference_mode()
    def _infer_single(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_tensor, pad_info, resized_w, resized_h = self._prepare_input(image)
        pred_depth, _, _ = self.model.inference({"input": input_tensor})
        pred_depth = pred_depth.squeeze()

        pad_top, pad_bottom, pad_left, pad_right = pad_info
        pred_depth = pred_depth[pad_top : pred_depth.shape[0] - pad_bottom, pad_left : pred_depth.shape[1] - pad_right]
        pred_depth = pred_depth.clamp_min(0.5)
        pred_depth = F.interpolate(pred_depth[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze()

        focal_px = self._intrinsics_to_focal_px(intrinsics, resized_w, resized_h)
        pred_depth = torch.clamp(pred_depth * (focal_px / 1000.0), min=0.0, max=800.0) # 300.0
        return pred_depth

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
