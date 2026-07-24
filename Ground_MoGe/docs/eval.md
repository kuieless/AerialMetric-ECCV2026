# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Donwload the processed datasets from [Huggingface Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and put them in the `data/eval` directory, using `huggingface-cli`:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval --local-dir-use-symlinks False
```

Then unzip the downloaded files:

```bash
cd data/eval  
unzip '*.zip'
# rm *.zip # if you don't keep the zip files
```

## Configuration

See [`configs/eval/all_benchmarks.json`](../configs/eval/all_benchmarks.json) for an example of evaluation configurations on all benchmarks. You can modify this file to evaluate on different benchmarks or different baselines.

For metric-depth-only evaluation on MoGe benchmarks, use [`configs/eval/metric_benchmarks.json`](../configs/eval/metric_benchmarks.json).

For custom scene datasets, use [`configs/eval/custom_metric_depth_example.json`](../configs/eval/custom_metric_depth_example.json) or [`configs/eval/metric_benchmarks_plus_custom_example.json`](../configs/eval/metric_benchmarks_plus_custom_example.json).

Custom loaders supported in config:
- `loader: "wild"`: each scene contains `image/`, `depth/`, and `metadata_full.csv`
- `loader: "oblique"`: each scene contains `rgbs/` and `depth/`; intrinsics can be loaded from a merged CSV via:
  - `oblique_intrinsics_csv`: optional path to csv. If omitted, loader auto-searches:
    - `MoGe/assets/metadata/oblique/final_merged.csv` (recommended)
    - `<oblique_root>/final_merged.csv`
    - `MoGe/final_merged.csv` (legacy fallback)
  - `oblique_scene_key`: scene-name column, default `Scene_Name`
  - `oblique_image_key`: image-name column, default `Renamed_Image`
- `loader: "bench"`: each scene contains `image/` and `depth/`; intrinsics are selected by image resolution:
  - `1358x910` -> `fx,fy,cx,cy = 920.7998657, 920.7998657, 671.032876, 449.90414`
  - `1022x672` -> `fx,fy,cx,cy = 1012.333801, 1012.333801, 505.011284, 342.060631`
  - If your bench data has other resolutions, set `bench_intrinsics_by_resolution` in config:
    - Example: `"bench_intrinsics_by_resolution": {"1920x1080": [fx, fy, cx, cy]}`

All datasets now use the same metric-depth keys: `absrel`, `rmse`, `delta1`, `delta2`, where:
- `delta1`: threshold = `1.25`
- `delta2`: threshold = `1.25^2`

For custom scene datasets, output JSON also includes grouped statistics:
- `"<Benchmark>_by_group"`: metrics averaged by scene grouping.
  - `Oblique`: grouped as `Rural/Natural/City/Factory/Uncategorized`.
  - `Bench` and `Wild`: grouped by scene folder name.
- `"<Benchmark>_group_count"`: sample count per group.
- `"<Benchmark>_group_mean"`: mean of group-level metrics.

## Baseline

Some examples of baselines are provided in [`baselines/`](../baselines/). Pass the path to the baseline model python code to the `--baseline` argument of the evaluation script. 

## Run Evaluation

Run the script [`moge/scripts/eval_baseline.py`](../moge/scripts/eval_baseline.py) for baseline checkpoints.
For example, 

```bash
# Evaluate MoGe on the 10 benchmarks
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/all_benchmarks.json --output eval_output/moge.json --pretrained Ruicheng/moge-vitl --resolution_level 9

# Evaluate Depth Anything V2 on the 10 benchmarks. (NOTE: affine disparity)
python moge/scripts/eval_baseline.py --baseline baselines/da_v2.py --config configs/eval/all_benchmarks.json --output eval_output/da_v2.json

# Evaluate UniDepthV2-L on metric benchmarks + custom datasets
python moge/scripts/eval_baseline.py --baseline baselines/unidepth.py --config configs/eval/metric_benchmarks_plus_custom_example.json --output eval_output/unidepth_v2.json --version v2 --backbone vitl14

# Evaluate Depth-Anything-3 Metric Large
python moge/scripts/eval_baseline.py --baseline baselines/da3_metric.py --config configs/eval/metric_benchmarks.json --output eval_output/da3_metric_large.json --model depth-anything/DA3METRIC-LARGE

# Evaluate MoGe-2 metric (ViT-L) on metric benchmarks + custom datasets
python moge/scripts/eval_baseline.py --baseline baselines/moge2_metric.py --config configs/eval/metric_benchmarks_plus_custom_example.json --output eval_output/moge2_metric_vitl.json --pretrained Ruicheng/moge-2-vitl --resolution_level 9

# Same command with periodic logging every 20 samples and log file
python moge/scripts/eval_baseline.py --baseline baselines/moge2_metric.py --config configs/eval/metric_benchmarks_plus_custom_example.json --output eval_output/moge2_metric_vitl.json --pretrained Ruicheng/moge-2-vitl --resolution_level 9 --log_interval 20 --log_level INFO --log_file eval_output/moge2_metric_vitl.log

# Compute all metrics (default is metric-depth-only)
python moge/scripts/eval_baseline.py --baseline baselines/moge2_metric.py --config configs/eval/metric_benchmarks_plus_custom_example.json --output eval_output/moge2_metric_vitl_all_metrics.json --pretrained Ruicheng/moge-2-vitl --resolution_level 9 --all_metrics
```

For the independent LoRA ground pipeline, use [`moge/scripts/eval_baselinelora.py`](../moge/scripts/eval_baselinelora.py):

```bash
python moge/scripts/eval_baselinelora.py \
  --baseline baselines/moge2_lora.py \
  --lora_config /path/to/config-lora-all.json \
  --lora_weight /path/to/lora_checkpoint.pt \
  --lora_rank 96 \
  --resolution_level 9 \
  --config configs/eval/ground_metric_benchmarks_local.json \
  --output eval_output_release/moge2_ground_lora.json \
  --oracle
```

The `--baselies` `--input` `--output` arguments are for the inference script. The rest arguments, e.g. `--pretrained` `--resolution_level`, are custormized for loading the baseline model.

Details of the arguments:

```
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH  Path to the baseline model python code.
  --config PATH    Path to the evaluation configurations. Defaults to
                   "configs/eval/all_benchmarks.json".
  --output PATH    Path to the output json file.
  --oracle         Use oracle mode for evaluation, i.e., use the GT intrinsics
                   input.
  --dump_pred      Dump predition results.
  --dump_gt        Dump ground truth.
  --metric_depth_only / --all_metrics
                   Compute only metric-depth metrics and skip all other
                   metrics.
  --log_interval INTEGER
                   Print running metrics every N samples in each benchmark.
                   Set 0 to disable.
  --log_level [DEBUG|INFO|WARNING|ERROR]
                   Logging verbosity.
  --log_file PATH  Optional path to write logs.
  --help           Show this message and exit.
```



## Wrap a Customized Baseline

Wrap any baseline method with [`moge.test.baseline.MGEBaselineInterface`](../moge/test/baseline.py).
See [`baselines/`](../baselines/) for more examples.

It is a good idea to check the correctness of the baseline implementation by running inference on a small set of images via [`moge/scripts/infer_baselines.py`](../moge/scripts/infer_baselines.py):

```base
python moge/scripts/infer_baselines.py --baseline baselines/moge.py --input example_images/ --output infer_outupt/moge --pretrained Ruicheng/moge-vitl --maps --ply
```
