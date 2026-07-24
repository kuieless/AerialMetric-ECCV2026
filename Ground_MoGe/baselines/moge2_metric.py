# Reference: https://github.com/microsoft/MoGe
from pathlib import Path
from typing import Dict, Optional

import click
import torch
import utils3d

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    """
    MoGe-2 metric baseline wrapper.

    This wrapper always uses MoGe v2 outputs (`depth_metric`, `points_metric`).
    """

    def __init__(
        self,
        num_tokens: Optional[int],
        resolution_level: int,
        pretrained_model_name_or_path: str,
        checkpoint_path: str,
        use_fp16: bool,
        device: str = "cuda:0",
    ):
        from moge.model.v2 import MoGeModel

        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = int(resolution_level)
        self.use_fp16 = bool(use_fp16)

        model_source = self._resolve_model_source(checkpoint_path, pretrained_model_name_or_path)
        self.model = MoGeModel.from_pretrained(model_source).to(self.device).eval()

    def _resolve_model_source(self, checkpoint_path: str, pretrained_model_name_or_path: str) -> str:
        if checkpoint_path:
            checkpoint = Path(checkpoint_path).expanduser()
            if not checkpoint.exists():
                raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint}")
            return str(checkpoint)
        return pretrained_model_name_or_path

    @click.command()
    @click.option("--num_tokens", type=int, default=None, help="Optional number of base ViT tokens.")
    @click.option("--resolution_level", type=int, default=9, help="MoGe2 resolution level [0..9].")
    @click.option(
        "--pretrained",
        "pretrained_model_name_or_path",
        type=str,
        default="Ruicheng/moge-2-vitl",
        help="HuggingFace model id used when --checkpoint is not set.",
    )
    @click.option(
        "--checkpoint",
        "checkpoint_path",
        type=click.Path(),
        default="",
        help="Optional local checkpoint path. If set, load this checkpoint instead of --pretrained.",
    )
    @click.option("--fp16", "use_fp16", is_flag=True, help="Enable mixed precision inference.")
    @click.option("--device", type=str, default="cuda:0", help="Device to use.")
    @staticmethod
    def load(
        num_tokens: Optional[int] = None,
        resolution_level: int = 9,
        pretrained_model_name_or_path: str = "Ruicheng/moge-2-vitl",
        checkpoint_path: str = "",
        use_fp16: bool = False,
        device: str = "cuda:0",
    ):
        return Baseline(
            num_tokens=num_tokens,
            resolution_level=resolution_level,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint_path=checkpoint_path,
            use_fp16=use_fp16,
            device=device,
        )

    def _infer_impl(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor],
        apply_mask: bool,
    ) -> Dict[str, torch.Tensor]:
        if intrinsics is not None:
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None

        output = self.model.infer(
            image,
            fov_x=fov_x,
            resolution_level=self.resolution_level,
            num_tokens=self.num_tokens,
            apply_mask=apply_mask,
            use_fp16=self.use_fp16,
        )

        result = {
            "points_metric": output["points"],
            "depth_metric": output["depth"],
            "intrinsics": output["intrinsics"],
        }
        if "normal" in output:
            result["normal"] = output["normal"]
        if "mask" in output:
            result["mask"] = output["mask"]
        return result

    @torch.inference_mode()
    def infer(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._infer_impl(image=image, intrinsics=intrinsics, apply_mask=True)

    @torch.inference_mode()
    def infer_for_evaluation(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # For fair evaluation we keep all pixels and let GT mask decide valid area.
        return self._infer_impl(image=image, intrinsics=intrinsics, apply_mask=False)
