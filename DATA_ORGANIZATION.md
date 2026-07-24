# Data Organization

This document describes the dataset layouts expected by the aerial inference
and evaluation scripts.

## 1. Standard Layout

Use this layout when `--intrinsics_mode` is `auto` or `none`.

```text
decoupled/
├── Cleaned_Dataset_Campus/
│   └── image/
│       └── SampleID.jpg
├── Cleaned_Dataset_Factory/
│   └── image/
├── Cleaned_Dataset_Farm/
│   └── image/
└── Cleaned_Dataset_Gress/
    └── image/

decoupled-norm/
└── SceneName/
    └── SampleID/
        ├── image.jpg
        ├── depth.npy
        └── meta.json

Oblique/
└── SceneName/
    └── rgbs/
        └── SampleID.jpg

Wild/
└── SceneName/
    └── rgbs/
        └── SampleID.jpg
    ├── depth/
    │   └── SampleID.npy
    └── depth_1k/
        └── SampleID.npy
```

For Decoupled, `--decoupled_csv_dir` should point to `decoupled` so the evaluator can
read:

```text
final_dataset_campus.csv
final_dataset_factory.csv
final_dataset_farm.csv
final_dataset_grass.csv
```

For Oblique and Wild, image and depth stems must match in the GT root.
For Wild, `depth/` keeps the original lower-resolution depth maps and
`depth_1k/` stores the upsampled copies aligned to the image resolution.

```text
Oblique/
└── SceneName/
    ├── rgbs/
    │   └── SampleID.jpg
    └── depth/
        └── SampleID.npy

Wild/
└── SceneName/
    ├── rgbs/
    │   └── SampleID.jpg
    ├── depth/
    │   └── SampleID.npy
    └── depth_1k/
        └── SampleID.npy
```

## 2. Norm-Style Layout

Use this layout when `--intrinsics_mode` is `load` for Decoupled and Oblique.
Wild stays on the standard layout because it does not provide `meta.json`.

```text
<decoupled_norm_root>/
└── SceneName/
    └── SampleID/
        ├── image.jpg
        ├── depth.npy
        └── meta.json

Oblique-norm/
└── SceneName/
    └── SampleID/
        ├── image.jpg
        ├── depth.npy
        └── meta.json
```

Each `meta.json` must contain an `intrinsics` matrix. The inference code reads
that matrix, converts it to `fov_x`, and passes it to the model.

On this workspace, the norm-style Decoupled root is `/data1/szq/Val/decoupled-norm`.
If your local dataset uses a different folder name, use that path instead. The
required structure is the same.
When you run the full aerial benchmark with `--intrinsics_mode load`, keep the
Wild input root on the standard layout so the code can still evaluate Wild with
`intrinsics_mode=auto`.

## 3. Example Commands

Standard layout:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config MoGe/configs/Final_train/config-lora-all.json \
  --decoupled_input /path/to/decoupled \
  --decoupled_gt /path/to/decoupled-norm \
  --decoupled_csv_dir /path/to/decoupled \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --wild_input /path/to/Wild \
  --wild_gt /path/to/Wild \
  --output_dir /path/to/eval_outputs \
  --gpu 0 \
  --resize 0 \
  --batch_size 1 \
  --sampling_ratio 0.1 \
  --intrinsics_mode none \
  --mask_mode none
```

Norm-style layout:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config MoGe/configs/Final_train/config-lora-all.json \
  --decoupled_input /path/to/<decoupled_norm_root> \
  --decoupled_gt /path/to/<decoupled_norm_root> \
  --decoupled_csv_dir /path/to/<decoupled_norm_root> \
  --oblique_input /path/to/Oblique-norm \
  --oblique_gt /path/to/Oblique-norm \
  --wild_input /path/to/Wild \
  --wild_gt /path/to/Wild \
  --output_dir /path/to/eval_outputs \
  --gpu 0 \
  --resize 0 \
  --batch_size 1 \
  --sampling_ratio 0.1 \
  --intrinsics_mode load \
  --mask_mode none
```

## 4. Common Checks

`Found 0 scenes to process` usually means the input root does not contain scene
directories with a supported image folder.

`Missing required intrinsics file` or `has no meta.json files` means a Decoupled or
Oblique input root does not use the norm-style layout required by
`--intrinsics_mode load`.

`No valid samples found` usually means image and depth stems do not match or GT
depth files are missing.

`No CUDA GPUs are available` means PyTorch cannot see the selected GPU. Check
`CUDA_VISIBLE_DEVICES` and `nvidia-smi`.
