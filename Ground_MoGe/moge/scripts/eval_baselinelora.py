import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import logging
import math
from numbers import Number
from typing import *
import importlib
import importlib.util

import click


OBLIQUE_SCENE_CATEGORIES = {
    "Rural": [
        "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output",
        "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled",
        "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled",
        "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled",
        "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output",
        "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled", "BC2", "BC1", "L1"
    ],
    "Natural": [
        "lewis-output", "park5", "park13", "park14", "park10", "park0",
        "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled",
        "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled",
        "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled",
        "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled",
        "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled",
        "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"
    ],
    "City": [
        "yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"
    ],
    "Factory": [
        "factory_scene_1", "factory_scene_2"
    ],
}


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.option('--metric_depth_only/--all_metrics', default=True, show_default=True, help='Compute only metric-depth metrics and skip all other metrics.')
@click.option('--eval_depth_min', type=float, default=None, help='Optional minimum GT depth (meters) for evaluation mask filtering.')
@click.option('--eval_depth_max', type=float, default=None, help='Optional maximum GT depth (meters) for evaluation mask filtering.')
@click.option('--log_interval', type=int, default=50, show_default=True, help='Print running metrics every N samples in each benchmark. Set 0 to disable.')
@click.option('--log_level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False), default='INFO', show_default=True, help='Logging verbosity.')
@click.option('--log_file', type=click.Path(), default='', help='Optional path to write logs.')
@click.pass_context
def main(
    ctx: click.Context,
    baseline_code_path: str,
    config_path: str,
    oracle_mode: bool,
    output_path: Union[str, Path],
    dump_pred: bool,
    dump_gt: bool,
    metric_depth_only: bool,
    eval_depth_min: Optional[float],
    eval_depth_max: Optional[float],
    log_interval: int,
    log_level: str,
    log_file: str,
):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from moge.test.baseline import MGEBaselineInterface
    from moge.test.dataloader import EvalDataLoaderPipeline
    from moge.test.scene_dataloader import SceneDepthEvalDataLoader
    from moge.test.metrics import compute_metrics
    from moge.utils.geometry_torch import intrinsics_to_fov
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module

    def _extract_scene_name(filename: str) -> str:
        normalized = str(filename).replace("\\", "/")
        if "/" in normalized:
            return normalized.split("/", 1)[0]
        stem = Path(normalized).stem
        return stem if stem else normalized

    def _classify_oblique_scene(scene_name: str) -> str:
        scene_name_l = scene_name.lower()
        for category, scene_patterns in OBLIQUE_SCENE_CATEGORIES.items():
            if any(pattern.lower() in scene_name_l for pattern in scene_patterns):
                return category
        return "Uncategorized"

    def _get_group_name(benchmark_cfg: Dict[str, Any], sample: Dict[str, Any]) -> Optional[str]:
        loader_name = str(benchmark_cfg.get("loader", "moge")).lower()
        scene_name = _extract_scene_name(sample.get("filename", ""))

        if loader_name == "oblique":
            return _classify_oblique_scene(scene_name)
        if loader_name in {"bench", "wild"}:
            return scene_name if scene_name else "Uncategorized"
        return None

    def setup_logger(level_name: str, file_path: str = '') -> logging.Logger:
        logger = logging.getLogger('moge.eval')
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(getattr(logging, level_name.upper()))

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file_path:
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path_obj, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _extract_numeric_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
        flat = flatten_nested_dict(metrics_dict)
        numeric_items: Dict[str, float] = {}
        for k, v in flat.items():
            if isinstance(v, Number):
                value = float(v)
                if not math.isnan(value) and not math.isinf(value):
                    numeric_items[".".join(k)] = value
        return numeric_items

    def _metric_to_str(metrics_dict: Dict[str, Any], max_items: int = 6) -> str:
        numeric_items = _extract_numeric_metrics(metrics_dict)

        if not numeric_items:
            return 'no valid metrics'

        preferred_keys = [
            'depth_metric.absrel',
            'depth_metric.rmse',
            'depth_metric.delta1',
            'depth_metric.delta2',
            'depth_scale_invariant.absrel',
            'depth_scale_invariant.rmse',
            'depth_scale_invariant.delta1',
            'depth_scale_invariant.delta2',
            'depth_affine_invariant.absrel',
            'depth_affine_invariant.rmse',
            'depth_affine_invariant.delta1',
            'depth_affine_invariant.delta2',
            'disparity_affine_invariant.absrel',
            'disparity_affine_invariant.rmse',
            'disparity_affine_invariant.delta1',
            'disparity_affine_invariant.delta2',
            'points_metric.rel',
            'points_metric.delta1',
            'inference_time',
        ]

        selected: List[Tuple[str, float]] = []
        used_keys = set()

        for key in preferred_keys:
            if key in numeric_items and len(selected) < max_items:
                selected.append((key, numeric_items[key]))
                used_keys.add(key)

        for key in sorted(numeric_items.keys()):
            if key in used_keys or len(selected) >= max_items:
                continue
            selected.append((key, numeric_items[key]))

        return ', '.join(f'{k}={v:.4f}' for k, v in selected)

    def _safe_filename_component(name: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(name))
        safe = safe.strip('_')
        return safe or 'benchmark'

    def _to_tsv_cell(value: Any) -> str:
        if value is None:
            return ''
        return str(value).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

    def _write_bench_per_image_report(
        records: List[Dict[str, Any]],
        metric_keys: Set[str],
        benchmark_name: str,
        output_json_path: Union[str, Path],
    ) -> Path:
        output_json_path = Path(output_json_path)
        report_name = f'{output_json_path.stem}_{_safe_filename_component(benchmark_name)}_per_image.txt'
        report_path = output_json_path.parent / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)

        fixed_columns = ['filename', 'status', 'skip_reason', 'valid_pixels']
        metric_columns = sorted(metric_keys)
        columns = fixed_columns + metric_columns

        with report_path.open('w', encoding='utf-8') as f:
            f.write('\t'.join(columns) + '\n')
            for row in records:
                cells: List[str] = []
                for col in columns:
                    value = row.get(col, '')
                    if col in metric_keys:
                        if isinstance(value, Number):
                            value = f'{float(value):.6f}'
                        elif value is None:
                            value = ''
                    cells.append(_to_tsv_cell(value))
                f.write('\t'.join(cells) + '\n')

        return report_path

    if eval_depth_min is not None and eval_depth_max is not None and eval_depth_min > eval_depth_max:
        raise ValueError(f"Invalid depth range: eval_depth_min ({eval_depth_min}) > eval_depth_max ({eval_depth_max})")

    def _depth_range_str(depth_min: Optional[float], depth_max: Optional[float]) -> str:
        min_str = "-inf" if depth_min is None else f"{depth_min:g}"
        max_str = "+inf" if depth_max is None else f"{depth_max:g}"
        return f"[{min_str}, {max_str}]"

    logger = setup_logger(log_level, log_file)
    logger.info(
        'Eval start | baseline=%s | config=%s | output=%s | oracle=%s | dump_pred=%s | dump_gt=%s | metric_depth_only=%s | depth_range=%s | extra_args=%s',
        baseline_code_path,
        config_path,
        output_path,
        oracle_mode,
        dump_pred,
        dump_gt,
        metric_depth_only,
        _depth_range_str(eval_depth_min, eval_depth_max),
        ' '.join(ctx.args) if ctx.args else '(none)',
    )

    def build_data_loader(benchmark_cfg: Dict[str, Any]):
        loader_name = str(benchmark_cfg.get('loader', 'moge')).lower()
        loader_kwargs = {k: v for k, v in benchmark_cfg.items() if k != 'loader'}
        if loader_name in {'moge', 'default', 'benchmark'}:
            return EvalDataLoaderPipeline(**loader_kwargs)
        if loader_name in {'wild', 'oblique', 'bench'}:
            return SceneDepthEvalDataLoader(loader=loader_name, **loader_kwargs)
        raise ValueError(f'Unsupported loader "{loader_name}" in benchmark config.')

    def resolve_eval_settings(benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
        loader_name = str(benchmark_cfg.get('loader', 'moge')).lower()
        is_custom_loader = loader_name in {'wild', 'oblique', 'bench'}

        depth_min = eval_depth_min
        if depth_min is None:
            depth_min = benchmark_cfg.get('eval_depth_min', 1e-3 if is_custom_loader else None)

        depth_max = eval_depth_max
        if depth_max is None:
            depth_max = benchmark_cfg.get('eval_depth_max', 400.0 if is_custom_loader else None)

        if depth_min is not None:
            depth_min = float(depth_min)
        if depth_max is not None:
            depth_max = float(depth_max)
        if depth_min is not None and depth_max is not None and depth_min > depth_max:
            raise ValueError(
                f'Invalid depth range for benchmark "{loader_name}": '
                f'eval_depth_min ({depth_min}) > eval_depth_max ({depth_max}).'
            )

        require_pred_valid = bool(benchmark_cfg.get('require_pred_valid', is_custom_loader))
        clip_pred_to_eval_range = bool(benchmark_cfg.get('clip_pred_to_eval_range', is_custom_loader))
        min_valid_pixels = int(benchmark_cfg.get('min_valid_pixels', 10 if is_custom_loader else 1))
        min_valid_pixels = max(1, min_valid_pixels)

        return {
            'depth_min': depth_min,
            'depth_max': depth_max,
            'require_pred_valid': require_pred_valid,
            'clip_pred_to_eval_range': clip_pred_to_eval_range,
            'min_valid_pixels': min_valid_pixels,
        }
    
    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)
    logger.info('Baseline loaded | class=%s | device=%s', type(baseline).__name__, baseline.device)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info('Config loaded | benchmarks=%d', len(config))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    # Iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        benchmark_loader = str(benchmark_config.get('loader', 'moge')).lower()
        is_bench_loader = benchmark_loader == 'bench'
        metrics_list = []
        grouped_metrics: Dict[str, List[Dict[str, Any]]] = {}
        bench_per_image_records: List[Dict[str, Any]] = []
        bench_metric_keys: Set[str] = set()
        skipped_eval_mask = 0
        eval_settings = resolve_eval_settings(benchmark_config)
        with (
            build_data_loader(benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            logger.info(
                'Benchmark start | name=%s | loader=%s | samples=%d | depth_range=%s | require_pred_valid=%s | clip_pred=%s | min_valid_pixels=%d',
                benchmark_name,
                str(benchmark_config.get('loader', 'moge')).lower(),
                len(eval_data_pipe),
                _depth_range_str(eval_settings['depth_min'], eval_settings['depth_max']),
                eval_settings['require_pred_valid'],
                eval_settings['clip_pred_to_eval_range'],
                eval_settings['min_valid_pixels'],
            )
            # Iterate over the samples in the dataset
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                sample_filename = sample.get('filename', str(i))
                image = sample['image']
                gt_intrinsics = sample['intrinsics']
                need_cuda_sync = baseline.device.type == 'cuda' and torch.cuda.is_available()

                # Inference
                if need_cuda_sync:
                    torch.cuda.synchronize()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image)
                    if need_cuda_sync:
                        torch.cuda.synchronize()

                # Match reference custom-eval behavior for metric depth on custom datasets:
                # apply depth range filtering on GT, optionally require pred-valid pixels,
                # and skip samples with too few valid pixels.
                metrics_sample = sample
                depth_mask = sample['depth_mask']
                depth = sample['depth']
                range_mask = torch.ones_like(depth_mask, dtype=torch.bool)
                if eval_settings['depth_min'] is not None:
                    range_mask = range_mask & (depth >= eval_settings['depth_min'])
                if eval_settings['depth_max'] is not None:
                    range_mask = range_mask & (depth <= eval_settings['depth_max'])
                eval_mask = depth_mask & range_mask

                if 'depth_metric' in pred:
                    pred_depth_metric = pred['depth_metric']
                    if eval_settings['clip_pred_to_eval_range']:
                        if eval_settings['depth_min'] is not None or eval_settings['depth_max'] is not None:
                            clip_min = eval_settings['depth_min'] if eval_settings['depth_min'] is not None else float('-inf')
                            clip_max = eval_settings['depth_max'] if eval_settings['depth_max'] is not None else float('inf')
                            pred_depth_metric = pred_depth_metric.clamp(min=clip_min, max=clip_max)
                            pred['depth_metric'] = pred_depth_metric
                    pred_valid_mask = torch.isfinite(pred_depth_metric) & (pred_depth_metric > 0)
                    eval_mask = eval_mask & pred_valid_mask

                valid_pixels = int(eval_mask.sum().item())
                if valid_pixels < eval_settings['min_valid_pixels']:
                    skipped_eval_mask += 1
                    skip_reason = f'valid_pixels({valid_pixels})<{eval_settings["min_valid_pixels"]}'
                    logger.warning(
                        'Skip sample | benchmark=%s | sample=%s | reason=valid_pixels(%d)<%d | depth_range=%s',
                        benchmark_name,
                        sample_filename,
                        valid_pixels,
                        eval_settings['min_valid_pixels'],
                        _depth_range_str(eval_settings['depth_min'], eval_settings['depth_max']),
                    )
                    if is_bench_loader:
                        bench_per_image_records.append(
                            {
                                'filename': sample_filename,
                                'status': 'skipped',
                                'skip_reason': skip_reason,
                                'valid_pixels': valid_pixels,
                            }
                        )
                    pbar.update(1)
                    continue

                if not torch.equal(eval_mask, sample['depth_mask']):
                    metrics_sample = dict(sample)
                    metrics_sample['depth_mask'] = eval_mask

                # Compute metrics
                metrics, misc = compute_metrics(
                    pred,
                    metrics_sample,
                    vis=dump_pred or dump_gt,
                    metric_depth_only=metric_depth_only,
                )
                metrics['inference_time'] = timer.time
                metrics_list.append(metrics)
                if is_bench_loader:
                    sample_numeric_metrics = _extract_numeric_metrics(metrics)
                    bench_metric_keys.update(sample_numeric_metrics.keys())
                    bench_per_image_records.append(
                        {
                            'filename': sample_filename,
                            'status': 'ok',
                            'skip_reason': '',
                            'valid_pixels': valid_pixels,
                            **sample_numeric_metrics,
                        }
                    )

                group_name = _get_group_name(benchmark_config, sample)
                if group_name is not None:
                    grouped_metrics.setdefault(group_name, []).append(metrics)

                # Dump results
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)

                    if 'pred_points' in misc:
                        points = misc['pred_points'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    
                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if 'mask' in pred:
                            mask = pred['mask'].cpu().numpy()
                            depth = np.where(mask, depth, np.inf)
                        cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))

                    if 'mask' in pred:
                        mask = pred['mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                    if 'intrinsics' in pred:
                        intrinsics = pred['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)
                
                if dump_gt:
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if 'points' in sample:
                        points = sample['points']
                        cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'gt' / 'info.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    partial = {
                        **all_metrics,
                        benchmark_name: key_average(metrics_list),
                    }
                    if grouped_metrics:
                        partial[f"{benchmark_name}_by_group"] = {
                            group: key_average(group_metrics)
                            for group, group_metrics in sorted(grouped_metrics.items())
                            if group_metrics
                        }
                    Path(output_path).write_text(json.dumps(partial, indent=4))

                if log_interval > 0 and ((i + 1) % log_interval == 0 or i == len(eval_data_pipe) - 1):
                    running_metrics = key_average(metrics_list)
                    logger.info(
                        'Benchmark progress | name=%s | %d/%d | skipped=%d | %s',
                        benchmark_name,
                        i + 1,
                        len(eval_data_pipe),
                        skipped_eval_mask,
                        _metric_to_str(running_metrics),
                    )
                pbar.update(1)

            benchmark_overall = key_average(metrics_list)
            all_metrics[benchmark_name] = benchmark_overall
            logger.info(
                'Benchmark done | name=%s | processed=%d | skipped=%d | %s',
                benchmark_name,
                len(metrics_list),
                skipped_eval_mask,
                _metric_to_str(benchmark_overall, max_items=10),
            )
            if grouped_metrics:
                by_group = {
                    group: key_average(group_metrics)
                    for group, group_metrics in sorted(grouped_metrics.items())
                    if group_metrics
                }
                group_count = {
                    group: len(group_metrics)
                    for group, group_metrics in sorted(grouped_metrics.items())
                    if group_metrics
                }
                all_metrics[f"{benchmark_name}_by_group"] = by_group
                all_metrics[f"{benchmark_name}_group_count"] = group_count
                all_metrics[f"{benchmark_name}_group_mean"] = key_average(list(by_group.values())) if by_group else {}
                logger.info(
                    'Benchmark groups | name=%s | groups=%s',
                    benchmark_name,
                    ", ".join([f"{g}:{c}" for g, c in group_count.items()]),
                )
                for group_name, group_metric in by_group.items():
                    logger.info(
                        'Benchmark group done | benchmark=%s | group=%s | %s',
                        benchmark_name,
                        group_name,
                        _metric_to_str(group_metric, max_items=10),
                    )
            if is_bench_loader:
                bench_report_path = _write_bench_per_image_report(
                    records=bench_per_image_records,
                    metric_keys=bench_metric_keys,
                    benchmark_name=benchmark_name,
                    output_json_path=output_path,
                )
                logger.info(
                    'Bench per-image report saved | benchmark=%s | rows=%d | file=%s',
                    benchmark_name,
                    len(bench_per_image_records),
                    bench_report_path,
                )

    # Save final results
    overall_benchmark_metrics = [all_metrics[name] for name in config.keys() if name in all_metrics]
    all_metrics['mean'] = key_average(overall_benchmark_metrics) if overall_benchmark_metrics else {}
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))
    logger.info('Eval done | output=%s', output_path)
    logger.info('Eval mean | %s', _metric_to_str(all_metrics['mean'], max_items=10))


if __name__ == '__main__':
    main()
