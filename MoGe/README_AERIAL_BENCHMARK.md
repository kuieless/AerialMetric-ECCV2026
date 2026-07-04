# Aerial Benchmark

This repository provides a unified inference and evaluation entry point for aerial metric depth benchmark experiments based on MoGe.

The recommended public interface is:

```text
moge/scripts/code-final/aerial_eval_cli.py
```

| Model type | Description |
|---|---|
| `Moge2-Aerial` | LoRA checkpoint with rank 96 and alpha 192 |

## Environment

Create or activate the project environment before running evaluation.

Example:

```bash
conda activate moge310
```

The CLI should be launched from the repository environment where MoGe, PyTorch, OpenCV, PEFT, and the project dependencies are available.

## Pipeline

The CLI runs the complete benchmark pipeline:

```text
input images
  -> inference
  -> extract nested depth.npy files
  -> evaluate Decoupled / Oblique / Wild for the selected model type
  -> optionally delete intermediate predictions
```

Output layout:

```text
<output_dir>/<model_type>/<checkpoint_name>/
  Infer/<Dataset>/<Scene>/<SampleID>/depth.npy
  Extracted/<Dataset>/<Scene>/<SampleID>.npy
  Extracted/<Dataset>/Eval_Report_*.txt
  Extracted/Wild/fov_analysis_details.csv
  run.log
```

If `--cleanup_intermediate` is used, the CLI removes `Infer/` and extracted `.npy` files after reports are generated. Text and CSV reports are kept.

For the released paper numbers, use the original LoRA evaluation chain:
Decoupled and Oblique use GT intrinsics from the norm-style inputs, Wild does
not use intrinsics, and masks are only applied in evaluation when explicitly enabled.

## Data Layout

The inference scanner supports these image layouts:

```text
Scene/SampleID/image.jpg
Scene/SampleID/image.png
Scene/image/*.jpg
Scene/images/*.jpg
Scene/rgb/*.jpg
Scene/rgbs/*.jpg
Scene/color/*.jpg
```

Decoupled evaluation supports:

```text
Decoupled GT:
  Scene/SampleID/depth.npy

Decoupled CSV metadata:
  final_dataset_campus.csv
  final_dataset_factory.csv
  final_dataset_farm.csv
  final_dataset_grass.csv

Optional mask root:

```text
Decoupled masks:
  Scene-mask/SampleID.png
```
```

If the CSV files are stored separately from the GT root, pass `--decoupled_csv_dir`.

Oblique and Wild GT roots are expected to contain scene-level depth folders:

```text
Oblique or Wild GT:
  Scene/depth/SampleID.npy
  Scene/depths/SampleID.npy
  Scene/depth/SampleID_depth.npy
```

For Wild, keep the original depth maps in `depth/` and store the upsampled
1k-aligned copies in `depth_1k/`.

Optional mask root:

```text
Oblique masks:
  Scene-mask/SampleID.png
```

Wild FoV evaluation also reads per-scene metadata when available:

```text
Wild GT:
  Scene/metadata_full.csv
```

## Common Options

| Option | Description |
|---|---|
| `--model_type` |  `Moge2-Aerial` |
| `--checkpoint` | Single checkpoint file |
| `--checkpoint_root` | Directory of `.pt` checkpoints |
| `--lora_config` | Required for LoRA variants |
| `--output_dir` | Root output directory |
| `--gpu` | CUDA device id exposed through `CUDA_VISIBLE_DEVICES` |
| `--batch_size` | Inference batch size. Decoupled with intrinsics requires `1` (mixed image sizes within scenes); Oblique and Wild can use `8` |
| `--resize` | Long-edge resize. `0` means original resolution |
| `--intrinsics_mode` | `auto`, `load`, or `none`. Wild always uses `auto` regardless of this setting |
| `--mask_mode` | `load` reads PNG masks for Decoupled/Oblique and excludes white pixels from scoring. Ignored for Wild |
| `--apply_mask` | Internal. Wild uses `True`, Decoupled/Oblique use `False`. Not exposed as a user-facing flag |
| `--cleanup_intermediate` | Delete large intermediate predictions after reports are generated |

Intrinsics modes:

| Mode | Behavior |
|---|---|
| `auto` | Use `meta.json` intrinsics when available |
| `load` | Require `meta.json` intrinsics for every image |
| `none` | Do not load or pass intrinsics |

## Full Benchmark Suite
```text
--decoupled_input / --decoupled_gt
--oblique_input / --oblique_gt
--wild_input / --wild_gt
```

The CLI processes all provided datasets in one run.

## Moge2-Aerial Evaluation Example

Run LoRA-96 on Decoupled, Oblique, and Wild using the released paper chain:

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n moge310 python \
  /home/szq/moge2/MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config /path/to/config-lora-all.json \
  --decoupled_input /path/to/decoupled \
  --decoupled_gt /path/to/decoupled-norm \
  --decoupled_csv_dir /path/to/decoupled \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --wild_input /path/to/Wild \
  --wild_gt /path/to/Wild \
  --output_dir /path/to/eval_outputs \
  --gpu 7 \
  --resize 0 \
  --batch_size 8 \
  --intrinsics_mode load \
  --cleanup_intermediate
```

Use `--mask_mode load` only for the optional internal mask-aware analysis.

> **Batch size note:** When Decoupled is included in the run, use `--batch_size 1`.
> The Decoupled dataset contains images of varying resolutions within the same
> scene, and batch padding interferes with intrinsics-based inference. When
> running only Oblique and/or Wild, `--batch_size 8` is safe.



Useful checkpoint filters:

| Option | Description |
|---|---|
| `--step_interval 1000` | Evaluate checkpoints whose step is divisible by 1000 |
| `--exact_steps 1000,2400` | Always include these steps |
| `--min_step 1000` | Minimum checkpoint step |
| `--max_step 10000` | Maximum checkpoint step |
| `--include_latest` | Include `latest*.pt` |
| `--exclude_named_checkpoints` | Ignore non-numeric checkpoint names |

## Notes For Reproducibility

Use the same settings when comparing with previous results:

```text
model_type
checkpoint
batch_size
resize
intrinsics_mode
input dataset root
GT dataset root
Decoupled CSV directory
evaluation script version
```

`batch_size` can affect predictions because images in the same batch are padded to the maximum height and width in that batch. To reproduce old LoRA runs, use `--batch_size 8` when GPU memory allows. **Exception:** When evaluating Decoupled (Bench) with `--intrinsics_mode load`, use `--batch_size 1`. The Decoupled dataset contains images of varying resolutions within the same scene, and batch padding interferes with intrinsics-based inference. Oblique and Wild are not affected.

Use `--mask_mode load` only when you intentionally want the optional internal mask-aware analysis.
