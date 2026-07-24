# Reference: https://github.com/isl-org/ZoeDepth
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
        variant: str,
        checkpoint_path: str,
        pretrained_resource: str,
        pad_input: bool,
        flip_aug: bool,
        device: str,
    ):
        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        self.device = torch.device(device)
        self.pad_input = pad_input
        self.flip_aug = flip_aug

        resolved_resource = pretrained_resource.strip() if pretrained_resource else ""
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path and ckpt_path.exists():
            resolved_resource = f"local::{str(ckpt_path.resolve())}"

        variant = variant.lower()
        cfg_kwargs = {}
        # Important: do not pass empty pretrained_resource, otherwise it overrides
        # config defaults and disables loading pretrained ZoeDepth weights.
        if resolved_resource:
            cfg_kwargs["pretrained_resource"] = resolved_resource

        if variant == "nk":
            config = get_config("zoedepth_nk", "infer", **cfg_kwargs)
        elif variant == "n":
            config = get_config("zoedepth", "infer", **cfg_kwargs)
        elif variant == "k":
            config = get_config("zoedepth", "infer", config_version="kitti", **cfg_kwargs)
        else:
            raise ValueError(f'Unsupported ZoeDepth variant "{variant}".')

        self.model = build_model(config).to(self.device).eval()

    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../ZoeDepth", help="Path to ZoeDepth repository.")
    @click.option("--variant", type=click.Choice(["nk", "n", "k"]), default="nk", help="Model variant. nk is the strongest multi-head model.")
    @click.option("--checkpoint", "checkpoint_path", type=click.Path(), default="", help="Optional local checkpoint path.")
    @click.option(
        "--pretrained_resource",
        type=str,
        default="",
        help='Optional pretrained resource string (e.g., "url::https://..." or "local::/path/to.ckpt"). Empty uses model default.',
    )
    @click.option("--pad_input/--no_pad_input", default=True, help="Enable input padding augmentation.")
    @click.option("--flip_aug/--no_flip_aug", default=True, help="Enable horizontal flip augmentation.")
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    @staticmethod
    def load(
        repo_path: str = "../ZoeDepth",
        variant: str = "nk",
        checkpoint_path: str = "",
        pretrained_resource: str = "",
        pad_input: bool = True,
        flip_aug: bool = True,
        device: str = "cuda",
    ):
        return Baseline(
            repo_path=repo_path,
            variant=variant,
            checkpoint_path=checkpoint_path,
            pretrained_resource=pretrained_resource,
            pad_input=pad_input,
            flip_aug=flip_aug,
            device=device,
        )

    @torch.inference_mode()
    def _infer_single(self, image: torch.Tensor) -> torch.Tensor:
        x = image.unsqueeze(0).to(self.device)
        depth = self.model.infer(x, pad_input=self.pad_input, with_flip_aug=self.flip_aug)
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]
        return depth.to(self.device).float()

    @torch.inference_mode()
    def infer(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if image.ndim == 3:
            depth = self._infer_single(image)
        else:
            depth = torch.stack([self._infer_single(image[i]) for i in range(image.shape[0])], dim=0)
        return {"depth_metric": depth}
