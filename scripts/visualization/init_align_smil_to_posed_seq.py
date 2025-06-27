#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/29/2024
#
# Distributed under terms of the MIT license.

"""Visualization of initial alignment of SMIL to posed segmentation and keypoints"""

import argparse
import os
from os import path as osp

import numpy as np

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def main(exp_dir, subj_name_list, num_betas=10, fetal_smpl_data_dict_path=None):
    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    num_col = 10
    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        num_betas=num_betas,
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )
    for i, name in enumerate(subj_name_list):
        # read data
        subj_dir = osp.join(exp_dir, "subj_spec", name, "init_posed")
        transl_seq = np.load(osp.join(subj_dir, "transl_seq_his.npy"))[-1]
        global_orient_seq = np.load(osp.join(subj_dir, "global_orient_seq_his.npy"))[-1]
        body_pose_seq = np.load(osp.join(subj_dir, "body_pose_seq_his.npy"))[-1]
        beta = np.load(osp.join(subj_dir, "beta_his.npy"))[-1]

        delta_transl = delta_x * (i % num_col) + delta_y * (i // num_col)
        this_smpl_seq = SMPLSequence(
            poses_body=body_pose_seq,
            smpl_layer=smpl_layer,
            betas=beta,
            trans=transl_seq + delta_transl[None, :],
            poses_root=global_orient_seq,
        )
        v.scene.add(this_smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--subj_name_list", type=str, default="")
    parser.add_argument("--fetal_smpl_data_dict_path", default=None, type=str)
    parser.add_argument("--num_betas", default=10, type=int)
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

    main(args.exp_dir, subj_name_list, args.num_betas, args.fetal_smpl_data_dict_path)
