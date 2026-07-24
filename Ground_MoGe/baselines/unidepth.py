# Reference: https://github.com/lpiccinelli-eth/UniDepth
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import click
import torch

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    def __init__(
        self,
        repo_path: str,
        version: str,
        backbone: str,
        pretrained_model_name_or_path: str,
        checkpoint_path: str,
        use_intrinsics: bool,
        use_fp16: bool,
        device: str,
    ):
        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from unidepth.models import UniDepthV1, UniDepthV2

        self.device = torch.device(device)
        self.version = version
        self.use_intrinsics = use_intrinsics
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        valid_backbones = {
            "v1": {"vitl14", "cnvnxtl"},
            "v2": {"vitl14", "vitb14", "vits14"},
        }
        if backbone not in valid_backbones[version]:
            raise ValueError(f"Invalid backbone '{backbone}' for UniDepth {version}.")

        model_cls = UniDepthV2 if version == "v2" else UniDepthV1
        model_id = pretrained_model_name_or_path or f"lpiccinelli/unidepth-{version}-{backbone}"

        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt and ckpt.exists():
            config_path = Path(repo_path) / "configs" / f"config_{version}_{backbone}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Cannot find UniDepth config file: {config_path}")
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)

            model = model_cls(config)
            state = torch.load(str(ckpt), map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            info = model.load_state_dict(state, strict=False)
            print(f"[UniDepth] loaded local checkpoint with missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
        else:
            model = model_cls.from_pretrained(model_id)

        self.model = model.to(self.device).eval()

    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../UniDepth", help="Path to UniDepth repository.")
    @click.option("--version", type=click.Choice(["v1", "v2"]), default="v2", help="UniDepth version.")
    @click.option("--backbone", type=click.Choice(["vitl14", "vitb14", "vits14", "cnvnxtl"]), default="vitl14", help="Backbone. For v2, use vitl14/vitb14/vits14.")
    @click.option("--pretrained", "pretrained_model_name_or_path", type=str, default="", help="HuggingFace model id or local model directory. Empty means auto by version/backbone.")
    @click.option("--checkpoint", "checkpoint_path", type=click.Path(), default="", help="Optional local checkpoint file path.")
    @click.option("--use_intrinsics/--no_use_intrinsics", default=True, help="Whether to use GT intrinsics as model input when available.")
    @click.option("--fp16", "use_fp16", is_flag=True, help="Use FP16 autocast for inference.")
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    @staticmethod
    def load(
        repo_path: str = "../UniDepth",
        version: str = "v2",
        backbone: str = "vitl14",
        pretrained_model_name_or_path: str = "",
        checkpoint_path: str = "",
        use_intrinsics: bool = True,
        use_fp16: bool = False,
        device: str = "cuda",
    ):
        return Baseline(
            repo_path=repo_path,
            version=version,
            backbone=backbone,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint_path=checkpoint_path,
            use_intrinsics=use_intrinsics,
            use_fp16=use_fp16,
            device=device,
        )

    def _to_pixel_intrinsics(self, intrinsics: torch.Tensor, width: int, height: int) -> torch.Tensor:
        intrinsics_px = intrinsics.clone()
        intrinsics_px[0, :] = intrinsics_px[0, :] * width
        intrinsics_px[1, :] = intrinsics_px[1, :] * height
        return intrinsics_px

    @torch.inference_mode()
    def _infer_single(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = image.shape[-2:]
        rgb = image.unsqueeze(0).to(self.device) * 255.0

        intrinsics_px = None
        if intrinsics is not None and self.use_intrinsics:
            intrinsics_px = self._to_pixel_intrinsics(intrinsics.to(self.device), w, h).unsqueeze(0)

        if self.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if self.version == "v2":
                    output = self.model.infer(rgb, camera=intrinsics_px if self.use_intrinsics else None, normalize=True)
                else:
                    output = self.model.infer(rgb, intrinsics=intrinsics_px if self.use_intrinsics else None)
        else:
            if self.version == "v2":
                output = self.model.infer(rgb, camera=intrinsics_px if self.use_intrinsics else None, normalize=True)
            else:
                output = self.model.infer(rgb, intrinsics=intrinsics_px if self.use_intrinsics else None)

        depth = output["depth"]
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
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
