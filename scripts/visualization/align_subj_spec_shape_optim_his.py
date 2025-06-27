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
import torch
from os import path as osp

import numpy as np

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import Meshes, Skeletons, SMPLSequence
from aitviewer.utils_fetal_smil import kpt_skeleton, str2color
from aitviewer.viewer import Viewer


def main(data_dir, exp_dir, name, frame_idx_list, step_idx):

    # read data
    subj_data_dir = osp.join(data_dir, name)
    kpt_seq = np.load(osp.join(subj_data_dir, "kpt_seq.npy"))

    # TODO(YL 11/05):: remove this
    # ijk -> xyz
    # kpt_seq = kpt_seq[..., [1, 0, 2]]

    segm_vertex_seq = np.load(
        osp.join(subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )
    segm_faces_seq = np.load(
        osp.join(subj_data_dir, "segm_faces_seq.npy"), allow_pickle=True
    )

    # read data
    subj_dir = osp.join(exp_dir, "subj_spec", name)
    pose_dir = osp.join(subj_dir, f"{step_idx}_posed")
    transl_seq_hist = np.load(osp.join(pose_dir, "transl_seq_his.npy"))
    global_orient_seq_hist = np.load(
        osp.join(pose_dir, "global_orient_seq_his.npy")
    )
    body_pose_seq_hist = np.load(osp.join(pose_dir, "body_pose_seq_his.npy"))
    kpt_pred_seq_hist = np.load(osp.join(pose_dir, "kpt_pred_seq_his.npy"))

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    num_col = 10
    delta_x = np.array([0.4, 0, 0])
    delta_y = np.array([0, 0, 0.4])

    # read subj spec shape for last step
    if step_idx == 1:
        last_unposed_folder_name = "init_unposed"
    else:
        last_unposed_folder_name = f"{step_idx - 1}_unposed"
    last_subj_spec_shape_path = osp.join(
        subj_dir, last_unposed_folder_name, "shape_his.npy"
    )
    subj_spec_shape = np.load(last_subj_spec_shape_path)[-1]

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        v_template=subj_spec_shape,
    )

    for i, f_idx in enumerate(frame_idx_list):
        delta_transl = delta_x * (i % num_col) + delta_y * (i // num_col)

        # smpl mesh sequence
        smpl_seq = SMPLSequence(
            poses_body=body_pose_seq_hist[:, f_idx],
            smpl_layer=smpl_layer,
            betas=torch.zeros(1, 10),
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

        # pred keypoints
        _color = (*str2color["tab_red"], 1.0)
        pred_skeleton_seq = Skeletons(
            kpt_pred_seq_hist[:, f_idx] + delta_transl[None, None, :],
            kpt_skeleton,
            gui_affine=False,
            color=_color,  # RGBA
            name="Skeleton",
        )
        v.scene.add(pred_skeleton_seq)

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
    parser.add_argument("--step_idx", type=int, default=1)
    args = parser.parse_args()

    frame_idx_list = [int(i) for i in args.frame_idx_list.split(",")]

    main(args.data_dir, args.exp_dir, args.subj_name, frame_idx_list, args.step_idx)
