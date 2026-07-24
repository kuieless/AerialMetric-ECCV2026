# Reference: https://github.com/apple/ml-depth-pro
import copy
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
        checkpoint_path: str,
        hf_repo_id: str,
        hf_filename: str,
        default_focal_px: float,
        use_fp16: bool,
        device: str,
    ):
        repo_path = os.path.abspath(repo_path)
        repo_src = os.path.join(repo_path, "src")
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)

        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        self.device = torch.device(device)
        self.default_focal_px = float(default_focal_px)

        resolved_ckpt = self._resolve_checkpoint(repo_path, checkpoint_path, hf_repo_id, hf_filename)
        config = copy.deepcopy(DEFAULT_MONODEPTH_CONFIG_DICT)
        config.checkpoint_uri = resolved_ckpt

        precision = torch.float16 if use_fp16 and self.device.type == "cuda" else torch.float32
        model, transform = depth_pro.create_model_and_transforms(config=config, device=self.device, precision=precision)
        self.model = model.eval().to(self.device)
        self.transform = transform

    def _resolve_checkpoint(
        self,
        repo_path: str,
        checkpoint_path: str,
        hf_repo_id: str,
        hf_filename: str,
    ) -> str:
        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                return str(ckpt)

        local_default = Path(repo_path) / "checkpoints" / "depth_pro.pt"
        if local_default.exists():
            return str(local_default)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required when local Depth Pro checkpoint is missing. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        return hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, repo_type="model")

    @staticmethod
    @click.command()
    @click.option("--repo", "repo_path", type=click.Path(), default="../ml-depth-pro", help="Path to ml-depth-pro repository.")
    @click.option("--checkpoint", "checkpoint_path", type=click.Path(), default="", help="Optional local checkpoint path.")
    @click.option("--hf_repo", "hf_repo_id", type=str, default="apple/DepthPro", help="HuggingFace repo id used when local checkpoint is missing.")
    @click.option("--hf_filename", type=str, default="depth_pro.pt", help="Checkpoint filename in HuggingFace repo.")
    @click.option("--default_focal_px", type=float, default=0.0, help="Fallback focal length in pixels. Set <=0 to let Depth Pro estimate focal length when intrinsics are unavailable.")
    @click.option("--fp16", "use_fp16", is_flag=True, help="Use FP16 inference when running on CUDA.")
    @click.option("--device", type=str, default="cuda", help="Device to use.")
    def load(
        repo_path: str = "../ml-depth-pro",
        checkpoint_path: str = "",
        hf_repo_id: str = "apple/DepthPro",
        hf_filename: str = "depth_pro.pt",
        default_focal_px: float = 0.0,
        use_fp16: bool = False,
        device: str = "cuda",
    ):
        return Baseline(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            default_focal_px=default_focal_px,
            use_fp16=use_fp16,
            device=device,
        )

    def _get_focal_px(self, intrinsics: Optional[torch.Tensor], width: int, height: int) -> Optional[float]:
        if intrinsics is not None:
            fx_px = float(intrinsics[0, 0].item() * width)
            fy_px = float(intrinsics[1, 1].item() * height)
            focal = 0.5 * (fx_px + fy_px)
            if focal > 1e-6:
                return focal
        if self.default_focal_px > 1e-6:
            return self.default_focal_px
        return None

    @torch.inference_mode()
    def _infer_single(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = image.shape[-2:]
        focal_px = self._get_focal_px(intrinsics, w, h)

        pil_img = TF.to_pil_image(image.detach().cpu().clamp(0, 1))
        model_input = self.transform(pil_img)
        prediction = self.model.infer(model_input, f_px=focal_px)
        depth = prediction["depth"]
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
