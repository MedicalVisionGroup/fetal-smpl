#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/20/2024
#
# Distributed under terms of the MIT license.

""" """

from functools import partial

import numpy as np
import torch
from einops import reduce

from pytorch3d.ops import knn_points
from robust_loss_pytorch import adaptive, general
from smplx.utils import to_tensor

from .vertex_keypoint_regressor import VerticeKeypointRegressor


def _compute_kpt_loss(keypoints, kpt_seq, scale_dist, f_loss_kpt):
    # compute joint regression loss on joints in fetal
    diff_kpt = keypoints - kpt_seq
    d_kpt = torch.sqrt((diff_kpt * scale_dist).pow(2).sum(dim=-1))
    loss_kpt = f_loss_kpt(d_kpt.view(-1, 1)).mean()
    return loss_kpt


def _compute_segm_loss(
    vertices,
    segm_vertex_seq,
    scale_dist,
    f_loss_smpl_to_segm,
    f_loss_segm_to_smpl,
    coeff_segm_chamfer,
):
    # chamfer distance (l2, for now k=1)
    d_smpl_to_segm, d_segm_to_smpl_list = (
        batch_chamfer_distance_two_sided_smpl_and_segm(
            vertices, segm_vertex_seq, norm=2, K=1, scale_dist=scale_dist
        )
    )
    loss_smpl_to_segm = (
        coeff_segm_chamfer * f_loss_smpl_to_segm(d_smpl_to_segm.view(-1, 1)).mean()
    )

    d_segm_to_smpl_list = [d[..., 0:1] for d in d_segm_to_smpl_list]  # list of [n, 1]
    loss_segm_to_smpl_per_t = torch.cat(
        [
            reduce(f_loss_segm_to_smpl(d), "n c -> ()", "mean", c=1)
            for d in d_segm_to_smpl_list
        ]
    )
    loss_segm_to_smpl = coeff_segm_chamfer * reduce(
        loss_segm_to_smpl_per_t, "t -> ()", "mean"
    )
    return loss_smpl_to_segm, loss_segm_to_smpl


def get_adaptive_loss_func(loss_mode: str, device):
    # robust loss for keypoint regression and chamfer distance
    # loss_mode = "geman_mcclure" or 'l2'
    kwargs = {
        "num_dims": 1,
        "float_dtype": torch.float32,
        "device": device,
        "scale_lo": 1.0,
        "scale_init": 1.0,
    }
    if loss_mode == "l2":
        f_robust_loss = partial(
            adaptive.AdaptiveLossFunction,
            alpha_lo=2.0,
            alpha_hi=2.0,
            **kwargs,
        )
    elif loss_mode == "geman_mcclure":
        raise NotImplementedError("Adaptive alpha only (0, 2)")
        f_robust_loss = partial(
            adaptive.AdaptiveLossFunction,
            alpha_lo=-2.0,
            alpha_hi=-2.0,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    return f_robust_loss


def get_general_loss_func(loss_mode: str, device):
    if loss_mode == "l2":
        f_robust_loss = partial(
            general.lossfun,
            alpha=torch.Tensor([2.0]).to(device),
            scale=torch.Tensor([1.0]).to(device),
        )
    elif loss_mode == "geman_mcclure":
        f_robust_loss = partial(
            general.lossfun,
            alpha=torch.Tensor([-2.0]).to(device),
            scale=torch.Tensor([1.0]).to(device),
        )
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    return f_robust_loss


def chamfer_distance(p1, p2):
    """Compute chamfer distance between two point sets
    and return the distance from p1 to p2

    Args:
        p1 (torch.tensor): (n, 3)
        p2 (torch.tensor): (m, 3)

    Returns:
        torch.tensor: (n,)
    """

    p1 = p1.unsqueeze(0)  # (1, n, 3)
    p2 = p2.unsqueeze(0)  # (1, m, 3)

    # re-implmentation with pytorch3d
    dist_p1_to_p2, _, _ = knn_points(p1, p2, norm=2, K=1)  # (1, n, 1)

    return dist_p1_to_p2


def batch_chamfer_distance_two_sided_smpl_and_segm(
    p_smpl: torch.Tensor,
    p_segm: list[torch.Tensor],
    norm: int = 2,
    K: int = 1,
    scale_dist: float = 1,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Calculate chamfer distance between smpl vertices and segmentation vertices.
    Return p1 to p2 and p2 to p1 distances.

    Args:
        p_smpl (torch.Tensor): (b, n=6890, 3)
        p_segm (list[torch.Tensor]): [(m_i, 3)]
        norm (int, optional): [description]. Defaults to 2.
        K (int, optional): [description]. Defaults to 1. cannot be larger than any n_points.
        scale_dist (float, optional): Factor to scale up distance.
            Avoid square on very small values. Defaults to 1.

    Returns:
        dist_smpl_to_segm (torch.Tensor): (b, n, K)
        dist_semg_to_smpl_list (list[torch.Tensor]): [(m_i, K)]
    """
    assert p_smpl.shape[1] == 6890
    batch_size = p_smpl.shape[0]
    device = p_smpl.device
    assert len(p_segm) == batch_size

    # pad segm points to the same length
    n_segm_point_list = [p_segm_i.shape[0] for p_segm_i in p_segm]
    max_n_segm_point = max(n_segm_point_list)

    p_segm_padded = []
    for p_segm_i in p_segm:
        zero_pad = torch.zeros([max_n_segm_point - p_segm_i.shape[0], 3], device=device)
        p_segm_padded.append(torch.cat([p_segm_i, zero_pad], dim=0))
    p_segm_padded = torch.stack(p_segm_padded, dim=0)

    # length: smpl and segm
    len_smpl = torch.repeat_interleave(torch.tensor([6890], device=device), batch_size)
    len_segm = torch.tensor(n_segm_point_list, device=device)

    # scale points to avoid very small values
    p_smpl = p_smpl * scale_dist
    p_segm_padded = p_segm_padded * scale_dist

    # chamfer: smpl to segm
    dist_smpl_to_segm, _, _ = knn_points(
        p1=p_smpl,
        p2=p_segm_padded,
        lengths1=len_smpl,
        lengths2=len_segm,
        norm=norm,
        K=K,
    )  # (b, n, K)

    # chamfer: segm to smpl
    dist_segm_to_smpl, _, _ = knn_points(
        p1=p_segm_padded,
        p2=p_smpl,
        lengths1=len_segm,
        lengths2=len_smpl,
        norm=norm,
        K=K,
    )  # (b, m_max, K)
    dist_semg_to_smpl_list = [
        d[:n] for d, n in zip(dist_segm_to_smpl, n_segm_point_list)
    ]  # list of (m_i, K)

    return dist_smpl_to_segm, dist_semg_to_smpl_list


def set_smil_model_to_fetal_smpl(
    model, fetal_smpl_data_dict_path, subj_spec_shape=None
):
    model_num_betas = model.num_betas

    # load data dict of fetal smpl model
    data_dict = np.load(fetal_smpl_data_dict_path, allow_pickle=True).item()
    mean_shape = data_dict["mean_shape"]
    shape_blend_shape = data_dict["shape_blend_shape"]
    pose_blend_shape = data_dict["pose_blend_shape"]
    J_regressor_kpt = data_dict["J_regressor_kpt"]

    if subj_spec_shape is None:
        # set mean shape
        assert model.v_template.shape == mean_shape.shape
        dtype = model.v_template.dtype
        del model.v_template
        model.register_buffer("v_template", to_tensor(mean_shape, dtype=dtype))
    else:
        # set mean shape with subject specific shape
        assert model.v_template.shape == subj_spec_shape.shape
        dtype = model.v_template.dtype
        del model.v_template
        model.register_buffer("v_template", to_tensor(subj_spec_shape, dtype=dtype))

    # set shape blend shape
    shape_blend_shape_trunc = shape_blend_shape[..., :model_num_betas]
    assert model.shapedirs.shape == shape_blend_shape_trunc.shape
    dtype = model.shapedirs.dtype
    del model.shapedirs
    model.register_buffer("shapedirs", to_tensor(shape_blend_shape_trunc, dtype=dtype))

    # set pose blend shape
    assert model.posedirs.shape == pose_blend_shape.shape
    dtype = model.posedirs.dtype
    del model.posedirs
    model.register_buffer("posedirs", to_tensor(pose_blend_shape, dtype=dtype))

    # set J_regressor_kpt
    if model.vertex_keypoint_regressor is not None:
        new_vertex_keypoint_regressor = VerticeKeypointRegressor(
            J_regressor=model.J_regressor,
            vertex_ids=model.vertex_ids,
            J_regressor_kpt=J_regressor_kpt,
        )
        del model.vertex_keypoint_regressor
        model.vertex_keypoint_regressor = new_vertex_keypoint_regressor


def resize_smpl_model(model, scale_factor: float):
    """Resize smpl model linearly with scale_factor.
    Linear multiply scale_factor to v_template, shapedirs, posedirs.
    """
    assert scale_factor > 0
    if scale_factor == 1:
        return

    # resize v_template
    if model.v_template is not None:
        model.v_template *= scale_factor

    # resize shapedirs
    if model.shapedirs is not None:
        model.shapedirs *= scale_factor

    # resize posedirs
    if model.posedirs is not None:
        model.posedirs *= scale_factor
