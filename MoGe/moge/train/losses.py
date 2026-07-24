

# from typing import *
# import math

# import torch
# import torch.nn.functional as F
# import utils3d

# from ..utils.geometry_torch import (
#     weighted_mean, 
#     harmonic_mean, 
#     geometric_mean,
#     normalized_view_plane_uv,
#     angle_diff_vec3
# )
# from ..utils.alignment import (
#     align_points_scale_z_shift, 
#     align_points_scale, 
#     align_points_scale_xyz_shift,
#     align_points_z_shift,
# )


# def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
#     if beta == 0:
#         return err
#     else:
#         return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


# def _truncate_loss(loss_map: torch.Tensor, mask: torch.Tensor, truncation_percent: float):
#     """
#     Helper function to discard the top X% of high-loss pixels/elements.
#     Args:
#         loss_map: [..., N] Raw loss values.
#         mask: [..., N] Boolean mask indicating valid elements.
#         truncation_percent: float, 0.0 to 1.0 (e.g., 0.2 means discard top 20%).
#     Returns:
#         Scalar mean loss of the kept elements.
#     """
#     if truncation_percent <= 0.0:
#         # Standard weighted mean if no truncation
#         if mask.sum() == 0: return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
#         return (loss_map * mask.float()).sum() / mask.float().sum()

#     # Flatten valid losses for sorting
#     valid_losses = loss_map[mask]
    
#     if valid_losses.numel() == 0:
#         return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)

#     # Determine the number of elements to keep
#     keep_ratio = 1.0 - truncation_percent
#     k = max(1, int(valid_losses.numel() * keep_ratio))
    
#     # Use topk with largest=False to get the k *smallest* losses (i.e. drop the large ones)
#     topk_values, _ = torch.topk(valid_losses, k=k, largest=False)
    
#     return topk_values.mean()


# def affine_invariant_global_loss(
#     pred_points: torch.Tensor, 
#     gt_points: torch.Tensor, 
#     align_resolution: int = 64, 
#     beta: float = 0.0, 
#     trunc: float = 1.0, 
#     sparsity_aware: bool = False,
#     truncation_percent: float = 0.0  # 🔥 Added parameter
# ):
#     device = pred_points.device

#     mask = torch.isfinite(gt_points).all(dim=-1)
#     gt_points = torch.where(mask[..., None], gt_points, 1)

#     # Align
#     # pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
#     pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
    
#     # 🔥 [Safety Fix] If the mask is empty after resizing, return 0 loss immediately
#     if lr_mask.sum() == 0:
#         print("⚠️ Warning: Empty mask encountered in alignment. Returning 0 loss.")
#         # Return zero loss, empty dict, and zero scale
#         return torch.tensor(0.0, device=device, requires_grad=True), {}, torch.zeros((batch_shape if 'batch_shape' in locals() else 1), device=device)
#     scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
#     valid = scale > 0
#     scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

#     pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

#     # Compute raw loss map (per pixel)
#     weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
#     weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
    
#     # Calculate per-pixel smooth L1 loss (sum over x,y,z)
#     raw_loss_map = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).sum(dim=-1) 
    
#     # 🔥 Apply Robust Truncation
#     # Only consider pixels that are both in the original mask AND valid after alignment
#     final_mask = mask & valid[..., None, None]
    
#     loss = _truncate_loss(raw_loss_map, final_mask, truncation_percent)

#     if sparsity_aware:
#         # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
#         sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
#         loss = loss / (sparsity + 1e-7)

#     # Metrics
#     err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]
#     misc = {
#         'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
#         'delta': weighted_mean((err < 1).float(), mask).item()
#     }

#     return loss, misc, scale.detach()


# def monitoring(points: torch.Tensor):
#     return {
#         'std': points.std().item(),
#     }


# def compute_anchor_sampling_weight(
#     points: torch.Tensor, 
#     mask: torch.Tensor, 
#     radius_2d: torch.Tensor, 
#     radius_3d: torch.Tensor, 
#     num_test: int = 64
# ) -> torch.Tensor:
#     # Importance sampling to balance the sampled probability of fine strutures.
#     # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
#     #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

#     height, width = points.shape[-3:-1]

#     pixel_i, pixel_j = torch.meshgrid(
#         torch.arange(height, device=points.device), 
#         torch.arange(width, device=points.device),
#         indexing='ij'
#     )
    
#     test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
#     test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
#     test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
#     test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
#     test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
#     test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
#     test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
#     test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

#     weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
#     weight = torch.where(mask, weight, 0)
#     weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
#     return weight


# def affine_invariant_local_loss(
#     pred_points: torch.Tensor, 
#     gt_points: torch.Tensor, 
#     focal: torch.Tensor, 
#     global_scale: torch.Tensor, 
#     level: Literal[4, 16, 64], 
#     align_resolution: int = 32, 
#     num_patches: int = 16, 
#     beta: float = 0.0, 
#     trunc: float = 1.0, 
#     sparsity_aware: bool = False
# ):
#     device, dtype = pred_points.device, pred_points.dtype
#     *batch_shape, height, width, _ = pred_points.shape
#     batch_size = math.prod(batch_shape)

#     gt_mask = torch.isfinite(gt_points).all(dim=-1)
#     gt_points = torch.where(gt_mask[..., None], gt_points, 1)
#     pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

#     # Sample patch anchor points indices [num_total_patches]
#     radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
#     radius_3d = 0.5 / level / focal * gt_points[..., 2]
#     anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
#     where_mask = torch.where(gt_mask)
#     # --- 修复开始 ---
#     # 1. 提取要采样的权重
#     valid_weights = anchor_sampling_weights[where_mask]

#     # 2. 检查是否为空或全0
#     if valid_weights.numel() == 0 or valid_weights.sum() <= 0:
#         # 如果没有有效的采样点，直接返回 0 loss，避免 CUDA 崩溃
#         # 注意：确保返回值格式与函数末尾一致，通常是 (loss, misc_dict)
#         print(f"⚠️ Warning: valid_weights is empty in affine_invariant_local_loss. Skipping batch.")
#         return torch.tensor(0.0, device=anchor_sampling_weights.device, requires_grad=True), {}

#     # 3. 如果正常，再进行采样
#     random_selection = torch.multinomial(valid_weights, num_patches * batch_size, replacement=True)
#     # random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
#     patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

#     # Get patch indices [num_total_patches, patch_h, patch_w]
#     patch_i, patch_j = torch.meshgrid(
#         torch.arange(-radius_2d, radius_2d + 1, device=device), 
#         torch.arange(-radius_2d, radius_2d + 1, device=device),
#         indexing='ij'
#     )
#     patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
#     patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
#     patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
#     # Get patch mask and gt patch points
#     gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
#     gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
#     gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
#     gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
#     patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
#     patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

#     # Pick only non-empty patches
#     MINIMUM_POINTS_PER_PATCH = 32
#     nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
#     num_nonempty_patches = nonempty[0].shape[0]
#     if num_nonempty_patches == 0:
#         return torch.tensor(0.0, dtype=dtype, device=device), {}
    
#     # Finalize all patch variables
#     patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
#     patch_mask = patch_mask[nonempty]                                       # [num_nonempty_patches, patch_h, patch_w]
#     gt_patch_points = gt_patch_points[nonempty]                             # [num_nonempty_patches, patch_h, patch_w, 3]
#     gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                       # [num_nonempty_patches]
#     gt_patch_anchor_points = gt_patch_anchor_points[nonempty]               # [num_nonempty_patches, 3]
#     pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
#     # Align patch points
#     pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
#     local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
#     if global_scale is not None:
#         scale_differ = local_scale / global_scale[patch_batch_idx]
#         patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
#     else:
#         patch_valid = local_scale > 0
#     local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
#     patch_mask &= patch_valid[:, None, None]

#     pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                  # [num_patches_nonempty, patch_h, patch_w, 3]
    
#     # Compute loss
#     gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
#     patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
#     loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]
    
#     if sparsity_aware:
#         # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
#         sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
#         loss = loss / (sparsity + 1e-7)
#     loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
#     loss = loss.reshape(batch_shape)
    
#     err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

#     # Record any scalar metric
#     misc = {
#         'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
#         'delta': weighted_mean((err < 1).float(), patch_mask).item()
#     }

#     return loss, misc


# def normal_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
#     device, dtype = points.device, points.dtype
#     height, width = points.shape[-3:-1]

#     mask = torch.isfinite(gt_points).all(dim=-1)
#     gt_points = torch.where(mask[..., None], gt_points, 1)

#     leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
#     upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
#     leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
#     downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
#     rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

#     gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
#     gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
#     gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
#     gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
#     gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

#     mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
#     mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
#     mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
#     mask_downxright = mask_leftdown & mask_rightup & mask_leftup
#     mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

#     MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

#     loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
#             + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
#             + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
#             + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

#     loss = loss.mean() / (4 * max(points.shape[-3:-1]))

#     return loss, {}


# def edge_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
#     device, dtype = points.device, points.dtype
#     height, width = points.shape[-3:-1]

#     mask = torch.isfinite(gt_points).all(dim=-1)
#     gt_points = torch.where(mask[..., None], gt_points, 1)

#     dx = points[..., :-1, :, :] - points[..., 1:, :, :]
#     dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
#     gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
#     gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

#     mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
#     mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

#     MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

#     loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
#     loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
#     loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

#     return loss, {}


# def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
#     loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
#     loss = loss.mean(dim=(-2, -1))
#     return loss, {}


# def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
#     loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
#     loss = loss.mean(dim=(-2, -1))
#     return loss, {}


# def metric_scale_loss(
#     scale_pred: torch.Tensor, 
#     scale_gt: torch.Tensor,
#     truncation_percent: float = 0.0  # 🔥 Added parameter
# ):
#     valid = scale_gt > 0
#     # Compute per-sample loss (do not average yet)
#     raw_loss = torch.where(valid, F.mse_loss(scale_pred.log(), torch.where(valid, scale_gt.log(), 0), reduction='none'), 0)
    
#     # 🔥 Apply Truncation
#     if truncation_percent > 0:
#         return _truncate_loss(raw_loss, valid, truncation_percent), {}
    
#     return raw_loss.mean(), {}


# def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
#     mask = torch.isfinite(gt_normal).all(dim=-1)
#     gt_normal = torch.where(mask[..., None], gt_normal, 1)

#     loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
#     return loss, {}
from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
)

# 🔥 [安全组件] 防止零向量在计算夹角、Normalize时产生 NaN
def _safe_vec(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(v)
    fallback[..., 0] = eps # 给 X 轴一个微小偏移，防止变成纯 0
    return torch.where(norm < eps, v + fallback, v)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def _truncate_loss(loss_map: torch.Tensor, mask: torch.Tensor, truncation_percent: float):
    if truncation_percent <= 0.0:
        if mask.sum() == 0: return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
        return (loss_map * mask.float()).sum() / mask.float().sum()

    valid_losses = loss_map[mask]
    
    if valid_losses.numel() == 0:
        return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)

    keep_ratio = 1.0 - truncation_percent
    k = max(1, int(valid_losses.numel() * keep_ratio))
    
    topk_values, _ = torch.topk(valid_losses, k=k, largest=False)
    return topk_values.mean()


def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False,
    truncation_percent: float = 0.0
):
    device = pred_points.device
    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
    
    if lr_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}, torch.zeros(1, device=device)
        
    scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
    valid = scale > 0
    scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # 🛡️ 强制钳位 gt_points，防止除以 0
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   
    
    raw_loss_map = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).sum(dim=-1) 
    final_mask = mask & valid[..., None, None]
    loss = _truncate_loss(raw_loss_map, final_mask, truncation_percent)

    if sparsity_aware:
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1)).clamp_min(1e-5)
        loss = loss / (sparsity + 1e-7)

    # 🛡️ 防止度量指标除以 0
    err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2].clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }

    return loss, misc, scale.detach()


def monitoring(points: torch.Tensor):
    return {'std': points.std().item()}


def compute_anchor_sampling_weight(
    points: torch.Tensor, mask: torch.Tensor, radius_2d: torch.Tensor, radius_3d: torch.Tensor, num_test: int = 64
) -> torch.Tensor:
    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(torch.arange(height, device=points.device), torch.arange(width, device=points.device), indexing='ij')
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)
    test_mask = test_mask & mask[..., test_i, test_j]
    test_points = points[..., test_i, test_j, :]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor, gt_points: torch.Tensor, focal: torch.Tensor, global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], align_resolution: int = 32, num_patches: int = 16, beta: float = 0.0, 
    trunc: float = 1.0, sparsity_aware: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)

    gt_mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(gt_mask[..., None], gt_points, 1)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

    # 🛡️ 保护 Focal 防止除以 0
    focal = focal.clamp_min(1e-5)
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    valid_weights = anchor_sampling_weights[where_mask]

    if valid_weights.numel() == 0 or valid_weights.sum() <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    random_selection = torch.multinomial(valid_weights, num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]

    patch_i, patch_j = torch.meshgrid(torch.arange(-radius_2d, radius_2d + 1, device=device), torch.arange(-radius_2d, radius_2d + 1, device=device), indexing='ij')
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]
    gt_patch_points = gt_patch_points[nonempty]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    
    if global_scale is not None:
        # 🛡️ 保护 global_scale，防止除以 0 产生 NaN
        scale_differ = local_scale / global_scale[patch_batch_idx].clamp_min(1e-6)
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale[patch_batch_idx] > 0)
    else:
        patch_valid = local_scale > 0
        
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]
    
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    # 🛡️ 保护 patch_weight 除以 0
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None].clamp_min(1e-5))
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))
    
    if sparsity_aware:
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1)).clamp_min(1e-5)
        loss = loss / (sparsity + 1e-7)
        
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    # 🛡️ 保护 metric err 的除法
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None].clamp_min(1e-5)

    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc


def normal_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    # 🛡️ 洗净零向量！防止计算夹角时底层 Normalize 产生 NaN
    upxleft, leftxdown, downxright, rightxup = map(_safe_vec, (upxleft, leftxdown, downxright, rightxup))
    gt_upxleft, gt_leftxdown, gt_downxright, gt_rightxup = map(_safe_vec, (gt_upxleft, gt_leftxdown, gt_downxright, gt_rightxup))

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))
    return loss, {}


def edge_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    # 🛡️ 安全处理零向量
    dx, dy = _safe_vec(dx), _safe_vec(dy)
    gt_dx, gt_dy = _safe_vec(gt_dx), _safe_vec(gt_dy)

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    return loss.mean(dim=(-2, -1)), {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    # 🛡️ BCE 的概率必须严格控制在 (0, 1) 之间，绝对不能等于 0 或 1，否则导数爆炸
    pred_mask_prob = pred_mask_prob.clamp(1e-6, 1.0 - 1e-6)
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    return loss.mean(dim=(-2, -1)), {}


def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor, truncation_percent: float = 0.0):
    valid = scale_gt > 0
    # 🛡️ 网络预测的 scale 极有可能是 0 或负数，取 log 瞬间变 NaN。加入 clamp_min(1e-6)
    safe_scale_pred = scale_pred.clamp(min=1e-6)
    safe_scale_gt = scale_gt.clamp(min=1e-6)
    
    raw_loss = torch.where(valid, F.mse_loss(safe_scale_pred.log(), torch.where(valid, safe_scale_gt.log(), 0.0), reduction='none'), 0.0)
    
    if truncation_percent > 0:
        return _truncate_loss(raw_loss, valid, truncation_percent), {}
    
    return raw_loss.mean(), {}


def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    # 🛡️ 防止纯 0 向量传入 angle_between
    pred_normal = _safe_vec(pred_normal)
    gt_normal = _safe_vec(gt_normal)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}