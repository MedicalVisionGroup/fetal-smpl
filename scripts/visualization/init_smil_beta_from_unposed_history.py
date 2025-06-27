#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/29/2024
#
# Distributed under terms of the MIT license.

""" """

import argparse
from os import path as osp

import numpy as np

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.utils_fetal_smil import str2color
from aitviewer.viewer import Viewer


def main(exp_dir, name, fetal_smpl_data_dict_path=None):
    # read data
    subj_dir = osp.join(exp_dir, "subj_spec", name)
    init_unposed_dir = osp.join(subj_dir, "init_unposed")

    segm_vertex_seq = np.load(
        osp.join(init_unposed_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )
    unposed_vertices_list = np.concatenate(segm_vertex_seq, axis=0)
    beta_his = np.load(osp.join(init_unposed_dir, "beta_his.npy"))

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    # smpl mesh sequence
    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        num_betas=beta_his.shape[1],
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )
    poses_body = np.zeros((beta_his.shape[0], smpl_layer.bm.NUM_BODY_JOINTS * 3))
    smpl_seq = SMPLSequence(
        poses_body=poses_body,
        smpl_layer=smpl_layer,
        betas=beta_his,
        trans=None,
        poses_root=None,
    )
    v.scene.add(smpl_seq)

    # unposed vertices
    ptc = PointClouds(
        points=unposed_vertices_list[None], color=(*str2color["yellow"], 0.25)
    )
    v.scene.add(ptc)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--subj_name", type=str, required=True)
    parser.add_argument("--fetal_smpl_data_dict_path", default=None, type=str)
    args = parser.parse_args()

    main(args.exp_dir, args.subj_name, args.fetal_smpl_data_dict_path)
