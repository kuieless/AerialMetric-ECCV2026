from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class SceneDepthEvalDataLoader:
    """
    Lightweight dataloader for scene-structured custom datasets.

    Supported layouts:
    - Wild:
      root/scene_x/
        image/*.jpg
        depth/*.npy
        metadata_full.csv
    - Oblique:
      root/scene_x/
        rgbs/*.jpg
        depth/*.npy
    - Bench:
      root/scene_x/
        image/*.jpg
        depth/*.npy
    """

    def __init__(
        self,
        path: str,
        loader: str,
        image_dir: str = "image",
        rgb_dir: str = "rgbs",
        depth_dir: str = "depth",
        metadata_file: str = "metadata_full.csv",
        oblique_intrinsics_csv: str = "",
        oblique_scene_key: str = "Scene_Name",
        oblique_image_key: str = "Renamed_Image",
        bench_intrinsics_by_resolution: Optional[Dict[str, Sequence[float]]] = None,
        depth_unit: Optional[float] = 1.0,
        subset: Optional[int] = None,
        default_focal_px: float = 1000.0,
        image_extensions: Optional[Sequence[str]] = None,
        **_: Dict,
    ):
        self.root = Path(path)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.loader = loader.lower()
        self.image_dir = image_dir
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.metadata_file = metadata_file
        self.oblique_intrinsics_csv = oblique_intrinsics_csv
        self.oblique_scene_key = oblique_scene_key
        self.oblique_image_key = oblique_image_key
        self.bench_intrinsics_by_resolution = self._parse_bench_intrinsics_map(bench_intrinsics_by_resolution)
        self.depth_unit = depth_unit
        self.default_focal_px = float(default_focal_px)
        self.image_extensions = tuple(image_extensions) if image_extensions else IMAGE_EXTENSIONS
        self._cursor = 0

        if self.loader not in {"wild", "oblique", "bench"}:
            raise ValueError(f'Unsupported loader "{loader}". Expected one of: wild, oblique, bench')

        self.oblique_intrinsics_index = {}
        if self.loader == "oblique":
            self.oblique_intrinsics_index = self._load_oblique_intrinsics_index()

        self.samples = self._build_samples()
        if subset is not None and subset > 1:
            self.samples = self.samples[::subset]

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {self.root} for loader={self.loader}")

    def __len__(self):
        return len(self.samples)

    def start(self):
        self._cursor = 0

    def stop(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get(self):
        if self._cursor >= len(self.samples):
            raise IndexError("Data loader is exhausted.")

        item = self.samples[self._cursor]
        self._cursor += 1

        image_np = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
        if image_np is None:
            raise RuntimeError(f'Failed to read image: {item["image_path"]}')
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_h, image_w = image_np.shape[:2]

        depth_np = np.load(item["depth_path"]).astype(np.float32)
        if depth_np.ndim > 2:
            depth_np = np.squeeze(depth_np)
        if depth_np.shape != (image_h, image_w):
            depth_np = cv2.resize(depth_np, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        if self.depth_unit is not None:
            depth_np = depth_np * float(self.depth_unit)

        depth_mask = np.isfinite(depth_np) & (depth_np > 0)
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.any(depth_mask):
            depth_mask = np.ones_like(depth_mask, dtype=bool)
            depth_np = np.ones_like(depth_np, dtype=np.float32)

        intrinsics = self._make_normalized_intrinsics(item, image_w, image_h)

        return {
            "filename": item["filename"],
            "image": torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1),
            "depth": torch.from_numpy(depth_np).float(),
            "depth_mask": torch.from_numpy(depth_mask).bool(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "is_metric": self.depth_unit is not None,
            "has_sharp_boundary": False,
        }

    def _build_samples(self) -> List[Dict]:
        if self.loader == "wild":
            return self._build_wild_samples()
        if self.loader == "bench":
            return self._build_bench_samples()
        return self._build_oblique_samples()

    def _build_wild_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for scene_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            depth_root = scene_dir / self.depth_dir
            image_root = scene_dir / self.image_dir
            if not image_root.exists():
                image_root = scene_dir / self.rgb_dir
            meta_path = scene_dir / self.metadata_file
            if not depth_root.exists() or not image_root.exists() or not meta_path.exists():
                continue

            with meta_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = row.get("filename_img") or row.get("image") or row.get("rgb")
                    depth_name = row.get("filename_npy") or row.get("depth") or row.get("filename_depth")
                    if not image_name:
                        continue
                    if not depth_name:
                        depth_name = f"{Path(image_name).stem}.npy"

                    image_path = image_root / image_name
                    depth_path = depth_root / depth_name
                    if not image_path.exists() or not depth_path.exists():
                        continue

                    intrinsics_params = self._read_intrinsics_from_row(row)
                    samples.append(
                        {
                            "filename": f"{scene_dir.name}/{Path(image_name).stem}",
                            "image_path": image_path,
                            "depth_path": depth_path,
                            "intrinsics_params": intrinsics_params,
                        }
                    )
        return samples

    def _build_oblique_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for scene_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            depth_root = scene_dir / self.depth_dir
            image_root = scene_dir / self.rgb_dir
            if not depth_root.exists() or not image_root.exists():
                continue

            image_by_stem = {}
            for ext in self.image_extensions:
                for img_path in image_root.glob(f"*{ext}"):
                    image_by_stem[img_path.stem] = img_path

            for depth_path in sorted(depth_root.glob("*.npy")):
                stem = depth_path.stem
                image_path = image_by_stem.get(stem)
                if image_path is None:
                    continue
                intrinsics_params = self._lookup_oblique_intrinsics(scene_dir.name, image_path.name)
                samples.append(
                    {
                        "filename": f"{scene_dir.name}/{stem}",
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "intrinsics_params": intrinsics_params,
                    }
                )
        return samples

    def _build_bench_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for scene_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            depth_root = scene_dir / self.depth_dir
            image_root = scene_dir / self.image_dir
            if not depth_root.exists() or not image_root.exists():
                continue

            image_by_stem = {}
            for ext in self.image_extensions:
                for img_path in image_root.glob(f"*{ext}"):
                    image_by_stem[img_path.stem] = img_path

            for depth_path in sorted(depth_root.glob("*.npy")):
                stem = depth_path.stem
                image_path = image_by_stem.get(stem)
                if image_path is None:
                    continue
                samples.append(
                    {
                        "filename": f"{scene_dir.name}/{stem}",
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "intrinsics_params": None,
                    }
                )
        return samples

    def _load_oblique_intrinsics_index(self) -> Dict[Tuple[str, str], Dict]:
        csv_path = self._resolve_oblique_intrinsics_csv()
        if csv_path is None:
            return {}

        index: Dict[Tuple[str, str], Dict] = {}
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene_name = str(row.get(self.oblique_scene_key, "")).strip()
                image_name = str(row.get(self.oblique_image_key, "")).strip()
                if not scene_name or not image_name:
                    continue

                intrinsics = self._read_oblique_intrinsics_from_row(row)
                if intrinsics is None:
                    continue

                key_scene = scene_name.lower()
                image_stem = Path(image_name).stem.lower()
                image_name_l = image_name.lower()
                index[(key_scene, image_name_l)] = intrinsics
                index[(key_scene, image_stem)] = intrinsics
        return index

    def _resolve_oblique_intrinsics_csv(self) -> Optional[Path]:
        if self.oblique_intrinsics_csv:
            path = Path(self.oblique_intrinsics_csv)
            if not path.exists():
                raise FileNotFoundError(f"oblique_intrinsics_csv does not exist: {path}")
            return path

        moge_root = Path(__file__).resolve().parents[2]
        default_candidates = [
            moge_root / "assets" / "metadata" / "oblique" / "final_merged.csv",
            self.root / "final_merged.csv",
            moge_root / "final_merged.csv",
        ]
        for cand in default_candidates:
            if cand.exists():
                return cand
        return None

    def _lookup_oblique_intrinsics(self, scene_name: str, image_name: str) -> Optional[Dict]:
        if not self.oblique_intrinsics_index:
            return None
        key_scene = scene_name.lower()
        key_name = image_name.lower()
        key_stem = Path(image_name).stem.lower()
        return self.oblique_intrinsics_index.get((key_scene, key_name)) or self.oblique_intrinsics_index.get((key_scene, key_stem))

    def _read_intrinsics_from_row(self, row: Dict[str, str]) -> Optional[Tuple[float, float, float, float, float, float]]:
        try:
            fx = float(row["fx"])
            fy = float(row["fy"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            width = float(row.get("width", 0))
            height = float(row.get("height", 0))
            return fx, fy, cx, cy, width, height
        except (KeyError, TypeError, ValueError):
            return None

    def _read_oblique_intrinsics_from_row(self, row: Dict[str, str]) -> Optional[Dict]:
        def _get_float(keys: Sequence[str]) -> Optional[float]:
            for k in keys:
                v = row.get(k, None)
                if v is None:
                    continue
                if isinstance(v, str):
                    v = v.strip()
                    if not v:
                        continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return None

        width = _get_float(["width", "Width", "image_width", "Image_Width"])
        height = _get_float(["height", "Height", "image_height", "Image_Height"])

        candidates: List[Tuple[float, float, float, float]] = []

        fx_new = _get_float(["FocalLength_New", "focal_new", "fx_new"])
        cx_new = _get_float(["Cx_New", "cx_new"])
        cy_new = _get_float(["Cy_New", "cy_new"])
        if fx_new is not None and cx_new is not None and cy_new is not None:
            candidates.append((fx_new, fx_new, cx_new, cy_new))

        fx_full = _get_float(["FocalLength(px)", "FocalLength", "focal", "fx"])
        cx_full = _get_float(["PP_x", "cx", "Cx"])
        cy_full = _get_float(["PP_y", "cy", "Cy"])
        if fx_full is not None and cx_full is not None and cy_full is not None:
            candidates.append((fx_full, fx_full, cx_full, cy_full))

        if not candidates:
            return None

        return {
            "candidates": candidates,
            "width": width,
            "height": height,
        }

    def _parse_bench_intrinsics_map(
        self, bench_intrinsics_by_resolution: Optional[Dict[str, Sequence[float]]]
    ) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
        default_map = {
            "1358x910": (920.7998657, 920.7998657, 671.032876, 449.90414),
            "1022x672": (1012.333801, 1012.333801, 505.011284, 342.060631),
        }
        src = bench_intrinsics_by_resolution or default_map
        parsed: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        for key, value in src.items():
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                raise ValueError(
                    f'Invalid bench intrinsics for "{key}". Expected [fx, fy, cx, cy], got: {value}'
                )
            if "x" not in str(key):
                raise ValueError(
                    f'Invalid bench intrinsics resolution key "{key}". Expected format "WIDTHxHEIGHT".'
                )
            w_str, h_str = str(key).lower().split("x", 1)
            try:
                w = int(w_str.strip())
                h = int(h_str.strip())
                fx, fy, cx, cy = [float(v) for v in value]
            except ValueError as exc:
                raise ValueError(
                    f'Invalid bench intrinsics entry "{key}": {value}'
                ) from exc
            parsed[(w, h)] = (fx, fy, cx, cy)
        return parsed

    def _get_bench_intrinsics_by_shape(
        self, image_w: int, image_h: int
    ) -> Optional[Tuple[float, float, float, float]]:
        return self.bench_intrinsics_by_resolution.get((image_w, image_h))

    def _make_normalized_intrinsics(self, item: Dict, image_w: int, image_h: int) -> np.ndarray:
        intrinsics_params = item.get("intrinsics_params")
        if isinstance(intrinsics_params, tuple):
            fx, fy, cx, cy, width, height = intrinsics_params
            width = width if width > 0 else float(image_w)
            height = height if height > 0 else float(image_h)
            sx, sy = image_w / width, image_h / height
            fx, fy = fx * sx, fy * sy
            cx, cy = cx * sx, cy * sy
            return np.array(
                [
                    [fx / image_w, 0.0, cx / image_w],
                    [0.0, fy / image_h, cy / image_h],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        if isinstance(intrinsics_params, dict):
            candidates = intrinsics_params.get("candidates", [])
            width = intrinsics_params.get("width", None)
            height = intrinsics_params.get("height", None)
            if candidates:
                if width is not None and height is not None and width > 0 and height > 0:
                    # Trust explicit calibration image size if provided.
                    fx, fy, cx, cy = candidates[0]
                    sx, sy = image_w / float(width), image_h / float(height)
                    fx, fy = fx * sx, fy * sy
                    cx, cy = cx * sx, cy * sy
                else:
                    # Choose the candidate whose principal point is closest to image center.
                    def center_score(v):
                        _, _, cx_v, cy_v = v
                        return abs(cx_v / image_w - 0.5) + abs(cy_v / image_h - 0.5)

                    fx, fy, cx, cy = min(candidates, key=center_score)

                return np.array(
                    [
                        [fx / image_w, 0.0, cx / image_w],
                        [0.0, fy / image_h, cy / image_h],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

        if self.loader == "bench":
            bench_intrinsics = self._get_bench_intrinsics_by_shape(image_w, image_h)
            if bench_intrinsics is None:
                supported = ", ".join(
                    [f"{w}x{h}" for (w, h) in sorted(self.bench_intrinsics_by_resolution.keys())]
                )
                raise ValueError(
                    f"Unknown bench image resolution {image_w}x{image_h}. "
                    f"Supported resolutions: {supported}. "
                    'You can override with "bench_intrinsics_by_resolution" in config.'
                )
            fx, fy, cx, cy = bench_intrinsics
            return np.array(
                [
                    [fx / image_w, 0.0, cx / image_w],
                    [0.0, fy / image_h, cy / image_h],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        # Fallback pseudo intrinsics for datasets without calibration (e.g., oblique).
        return np.array(
            [
                [self.default_focal_px / image_w, 0.0, 0.5],
                [0.0, self.default_focal_px / image_h, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
