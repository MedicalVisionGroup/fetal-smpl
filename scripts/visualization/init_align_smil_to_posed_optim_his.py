#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/29/2024
#
# Distributed under terms of the MIT license.

"""Visualize the optimization process of aligning SMIL to posed segmentation and keypoints"""

import argparse
import os
from os import path as osp

import numpy as np

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import Meshes, Skeletons, SMPLSequence
from aitviewer.utils_fetal_smil import kpt_skeleton, str2color
from aitviewer.viewer import Viewer


def main(data_dir, exp_dir, name, frame_idx_list, fetal_smpl_data_dict_path=None):
    # read data
    subj_data_dir = osp.join(data_dir, name)
    kpt_seq = np.load(osp.join(subj_data_dir, "kpt_seq.npy"))
    segm_vertex_seq = np.load(
        osp.join(subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )
    segm_faces_seq = np.load(
        osp.join(subj_data_dir, "segm_faces_seq.npy"), allow_pickle=True
    )

    subj_dir = osp.join(exp_dir, "subj_spec", name)
    init_pose_dir = osp.join(subj_dir, "init_posed")
    transl_seq_hist = np.load(osp.join(init_pose_dir, "transl_seq_his.npy"))
    global_orient_seq_hist = np.load(
        osp.join(init_pose_dir, "global_orient_seq_his.npy")
    )
    body_pose_seq_hist = np.load(osp.join(init_pose_dir, "body_pose_seq_his.npy"))
    beta_hist = np.load(osp.join(init_pose_dir, "beta_his.npy"))

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    num_col = 10
    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        num_betas=beta_hist.shape[1],
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )

    for i, f_idx in enumerate(frame_idx_list):
        delta_transl = delta_x * (i % num_col) + delta_y * (i // num_col)

        # smpl mesh sequence
        smpl_seq = SMPLSequence(
            poses_body=body_pose_seq_hist[:, f_idx],
            smpl_layer=smpl_layer,
            betas=beta_hist[:, 0],
            trans=transl_seq_hist[:, f_idx] + delta_transl[None, :],
            poses_root=global_orient_seq_hist[:, f_idx],
        )
        v.scene.add(smpl_seq)

        # keypoints
        _color = (*str2color["tab_blue"], 1.0)
        skeleton_seq = Skeletons(
            kpt_seq[f_idx][None, :] + delta_transl[None, None, :],
            kpt_skeleton,
            gui_affine=False,
            color=_color,  # RGBA
            name="Skeleton",
        )
        v.scene.add(skeleton_seq)

        # add mesh
        verts = segm_vertex_seq[f_idx] + delta_transl[None, :]
        faces = segm_faces_seq[f_idx]
        mesh_seq = Meshes(verts, faces, color=(0.5, 0.5, 0.5, 0.5), name="BodySegm")
        v.scene.add(mesh_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--subj_name", type=str, required=True)
    parser.add_argument("--frame_idx_list", type=str, required=True)
    parser.add_argument("--fetal_smpl_data_dict_path", default=None, type=str)
    args = parser.parse_args()

    frame_idx_list = [int(i) for i in args.frame_idx_list.split(",")]

    main(
        args.data_dir,
        args.exp_dir,
        args.subj_name,
        frame_idx_list,
        args.fetal_smpl_data_dict_path,
    )
