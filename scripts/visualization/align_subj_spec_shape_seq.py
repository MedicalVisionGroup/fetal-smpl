#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/29/2024
#
# Distributed under terms of the MIT license.

"""Visualization of alignment of subj specific shape to posed segmentation and keypoints"""

import argparse
import os
from os import path as osp

import numpy as np
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def main(exp_dir, subj_name_list, step_idx):
    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    num_col = 10
    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])

    for i, name in enumerate(subj_name_list):
        # read data
        exp_subj_dir = osp.join(exp_dir, "subj_spec", name)
        exp_subj_posed_dir = osp.join(exp_subj_dir, f"{step_idx}_posed")
        transl_seq = np.load(osp.join(exp_subj_posed_dir, "transl_seq_his.npy"))[-1]
        global_orient_seq = np.load(
            osp.join(exp_subj_posed_dir, "global_orient_seq_his.npy")
        )[-1]
        body_pose_seq = np.load(osp.join(exp_subj_posed_dir, "body_pose_seq_his.npy"))[
            -1
        ]

        # read subj spec shape for last step
        if step_idx == 1:
            last_unposed_folder_name = "init_unposed"
        else:
            last_unposed_folder_name = f"{step_idx - 1}_unposed"
        last_subj_spec_shape_path = osp.join(
            exp_subj_dir, last_unposed_folder_name, "shape_his.npy"
        )
        subj_spec_shape = np.load(last_subj_spec_shape_path)[-1]

        smpl_layer = SMPLLayer(
            model_type="smpl",
            gender="infant",
            v_template=subj_spec_shape,
        )

        # color option 1: rgb = (180, 149, 133)
        color1 = torch.tensor([180, 149, 133, 255]) / 255.0

        # color option 2:
        color2 = torch.tensor([0.9, 0.7, 0.6, 1.0])

        delta_transl = delta_x * (i % num_col) + delta_y * (i // num_col)
        this_smpl_seq = SMPLSequence(
            poses_body=body_pose_seq,
            smpl_layer=smpl_layer,
            betas=torch.zeros(1, 10),
            trans=transl_seq + delta_transl[None, :],
            poses_root=global_orient_seq,
            color=color1,
            is_rigged=False,
        )
        v.scene.add(this_smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--subj_name_list", type=str, default="")
    parser.add_argument("--step_idx", type=int, default=1)
    args = parser.parse_args()

    subj_spec_result_dir = osp.join(args.exp_dir, "subj_spec")
    all_avail_subj_name_list = sorted(os.listdir(subj_spec_result_dir))

    if len(args.subj_name_list) == 0:
        subj_name_list = all_avail_subj_name_list
    else:
        args.subj_name_list = args.subj_name_list.split(",")
        args.subj_name_list = [sn.strip() for sn in args.subj_name_list]
        assert all(sn in all_avail_subj_name_list for sn in args.subj_name_list)
        subj_name_list = args.subj_name_list

    main(args.exp_dir, subj_name_list, args.step_idx)
