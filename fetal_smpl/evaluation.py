#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/22/2024
#
# Distributed under terms of the MIT license.

""" """

import os
from os import path as osp

import numpy as np
import torch

from .utils import batch_chamfer_distance_two_sided_smpl_and_segm
from .vertex_keypoint_regressor import _kpt_name_list


def _to_np(x):
    return x.detach().cpu().numpy()


def evaluate_alignment_to_posed_segm_kpt(
    model,
    beta,
    body_pose,
    global_orient,
    transl,
    kpt_seq,
    segm_vertex_seq,
    save_dir,
    scale_dist=100,
):
    """Evaluation step after we optimize the alignment to posed segm and kpt.
    Assume the input model/variables are already optimized.

    Args:
        model (nn.Module): smplx model
        body_pose (torch.Tensor): [n, 69]
        global_orient (torch.Tensor): [n, 3]
        transl (torch.Tensor): [n, 3]
        kpt_seq (torch.Tensor): [n, 24, 3]
        segm_vertex_seq (torch.Tensor): list of [n_v, 3]
        save_dir (str): save dir
        scale_dist (int, optional): scale distance to avoid small value.
            Defaults to 100.
    """
    # run inference
    output = model(
        betas=beta,
        body_pose=body_pose,
        transl=transl,
        global_orient=global_orient,
        return_verts=True,
    )
    joints = output.joints
    keypoints = joints[:, -len(_kpt_name_list) :, :]
    vertices = output.vertices

    # keypoints:
    diff_kpt = keypoints - kpt_seq
    d_kpt = torch.sqrt((diff_kpt * scale_dist).pow(2).sum(dim=-1))
    d_kpt = _to_np(d_kpt) / scale_dist

    # segmentation: two sided chamfer distance
    d_smpl_to_segm, d_segm_to_smpl_list = (
        batch_chamfer_distance_two_sided_smpl_and_segm(
            vertices, segm_vertex_seq, norm=2, K=1, scale_dist=scale_dist
        )
    )
    d_smpl_to_segm = _to_np(d_smpl_to_segm) / scale_dist
    d_segm_to_smpl_list = [_to_np(d) / scale_dist for d in d_segm_to_smpl_list]
    d_segm_to_smpl_list = np.array(d_segm_to_smpl_list, dtype=object)

    # save metrics
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "d_kpt.npy"), d_kpt)
    np.save(osp.join(save_dir, "d_smpl_to_segm.npy"), d_smpl_to_segm)
    np.save(osp.join(save_dir, "d_segm_to_smpl_list.npy"), d_segm_to_smpl_list)
