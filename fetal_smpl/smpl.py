#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 06/26/2025
#
# Distributed under terms of the MIT license.

""" """

import os
import os.path as osp

import smplx

from .vertex_keypoint_regressor import VerticeKeypointRegressor


class SMPL(smplx.SMPL):
    def __init__(self, *args, **kwargs):
        # we assume all of them are possed through kwargs.
        J_regressor_kpt = kwargs.pop("J_regressor_kpt", None)
        return_kpt = kwargs.pop("return_kpt", True)

        vertex_ids = kwargs.get("vertex_ids", None)
        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = smplx.vertex_ids.vertex_ids["smplh"]
        self.vertex_ids = vertex_ids

        super(SMPL, self).__init__(*args, **kwargs)

        if return_kpt:
            self.vertex_keypoint_regressor = VerticeKeypointRegressor(
                J_regressor=self.J_regressor,
                vertex_ids=self.vertex_ids,
                J_regressor_kpt=J_regressor_kpt,
            )
        else:
            self.vertex_keypoint_regressor = None


def create(model_path: str, model_type: str = "smpl", **kwargs) -> SMPL:
    """Method for creating a model from a path and a model type

    Parameters
    ----------
    model_path: str
        Either the path to the model you wish to load or a folder,
        where each subfolder contains the differents types, i.e.:
        model_path:
        |
        |-- smpl
            |-- SMPL_FEMALE
            |-- SMPL_NEUTRAL
            |-- SMPL_MALE
            |-- SMPL_INFANT

    model_type: str, optional
        When model_path is a folder, then this parameter specifies  the
        type of model to be loaded
    **kwargs: dict
        Keyword arguments

    Returns
    -------
        body_model: nn.Module
            The PyTorch module that implements the corresponding body model
    Raises
    ------
        ValueError: In case the model type is not SMPL
    """

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split("_")[0].lower()

    if model_type.lower() == "smpl":
        return SMPL(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type {model_type}, exiting!")
