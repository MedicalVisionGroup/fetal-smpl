#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/23/2024
#
# Distributed under terms of the MIT license.

"""Regress keypoints from vertices"""

import typing as tp

import torch
import torch.nn as nn

from smplx.lbs import vertices2joints
from smplx.utils import to_np, to_tensor

# keypoint name list from fetal keypoint detection model
_kpt_name_list = [
    "ankle_r",
    "ankle_l",
    "knee_r",
    "knee_l",
    "bladder",
    "elbow_r",
    "elbow_l",
    "eye_r",
    "eye_l",
    "hip_r",
    "hip_l",
    "shoulder_r",
    "shoulder_l",
    "wrist_r",
    "wrist_l",
    "neck",
]

_kpt2smpl_joint = [8, 7, 5, 4, 0, 19, 18, 25, 26, 2, 1, 17, 16, 21, 20, 12]

# map felal keypoint to parent joint in SMPL
kpt2smpl_joint_parent_idx_map = [
    5,  # ankle_r <- knee_r
    4,  # ankle_l <- knee_l
    2,  # knee_r <- hip_r
    1,  # knee_l <- hip_l
    -1,  # bladder (root)
    17,  # elbow_r <- shoulder_r
    16,  # elbow_l <- shoulder_l
    15,  # eye_r <- head
    15,  # eye_l <- head
    0,  # hip_r <- bladder
    0,  # hip_l <- bladder
    14,  # shoulder_r <- collar_r
    13,  # shoulder_l <- collar_l
    19,  # wrist_r <- elbow_r
    18,  # wrist_l <- elbow_l
    9,  # neck <- spine3
]


class VerticeKeypointRegressor(nn.Module):
    def __init__(
        self,
        J_regressor: torch.Tensor,
        vertex_ids: tp.Dict[str, int],
        J_regressor_kpt: tp.Optional[torch.Tensor] = None,
    ):
        super(VerticeKeypointRegressor, self).__init__()

        # derive the initial value of J_regressor_kpt and J_regressor_kpt_support
        # from J_regressor and vertex_ids following SMPL's joint order
        J_regressor_kpt_init = []
        for idx_smpl_joint in _kpt2smpl_joint:
            if idx_smpl_joint == 25:  # eye_r
                r = torch.zeros(J_regressor.shape[1], dtype=J_regressor.dtype)
                r[vertex_ids["reye"]] = 1
                J_regressor_kpt_init.append(r)

            elif idx_smpl_joint == 26:  # eye_l
                r = torch.zeros(J_regressor.shape[1], dtype=J_regressor.dtype)
                r[vertex_ids["leye"]] = 1
                J_regressor_kpt_init.append(r)

            else:
                J_regressor_kpt_init.append(J_regressor[idx_smpl_joint])
        J_regressor_kpt_init = torch.stack(J_regressor_kpt_init, dim=0)  # (J_kpt, V)
        self.register_buffer("J_regressor_kpt_init", J_regressor_kpt_init)

        # support is the non-zero 0/1 mask of J_regressor_kpt
        J_regressor_kpt_support = J_regressor_kpt_init.ne(0).int()

        # for reye and leye, we add the support of ear and head
        # so that we can regress center of eye ball.
        kpt_support_head = J_regressor_kpt_support[15]
        J_regressor_kpt_support[7] = torch.max(
            J_regressor_kpt_support[7], kpt_support_head
        )
        J_regressor_kpt_support[7][vertex_ids["rear"]] = 1
        J_regressor_kpt_support[8] = torch.max(
            J_regressor_kpt_support[8], kpt_support_head
        )
        J_regressor_kpt_support[8][vertex_ids["lear"]] = 1

        self.register_buffer("J_regressor_kpt_support", J_regressor_kpt_support)

        # if args J_regressor_kpt is not None, we check if it is consistent
        # with J_regressor_kpt_support
        if J_regressor_kpt is not None:
            # all zero in support should be zero in J_regressor_kpt
            J_regressor_kpt = to_tensor(to_np(J_regressor_kpt), dtype=torch.float32)
            assert torch.all(
                J_regressor_kpt_init[self.J_regressor_kpt_support == 0] == 0
            )
            self.register_parameter(
                "J_regressor_kpt", torch.nn.Parameter(J_regressor_kpt)
            )
        else:
            self.register_parameter(
                "J_regressor_kpt", torch.nn.Parameter(J_regressor_kpt_init)
            )

    def forward(self, vertices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertices (torch.Tensor): (B, V, 3)

        Returns:
            torch.Tensor: (B, J_kpt, 3)
        """
        joints_kpt = vertices2joints(self.J_regressor_kpt, vertices)
        joints = torch.cat([joints, joints_kpt], dim=1)
        return joints
