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
import pickle
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from smplx.lbs import blend_shapes, lbs
from smplx.utils import Array, SMPLOutput, Struct, Tensor, to_np, to_tensor
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.vertex_joint_selector import VertexJointSelector

from .vertex_keypoint_regressor import VerticeKeypointRegressor


class SMPL(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self,
        model_path: str,
        kid_template_path: str = "",
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = "neutral",
        age: str = "adult",
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        J_regressor_kpt: Optional[Tensor] = None,
        return_kpt: bool = False,
        **kwargs,
    ) -> None:
        """SMPL model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        num_betas: int, optional
            Number of shape components to use
            (default = 10).
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        v_template: torch.tensor, optional
            The vertices of the template model
        J_regressor_kpt: torch.tensor, optional
            The J_regressor for the keypoint joints
        return_kpt: bool, optional
            If True, then the forward pass will return the keypoints
            (default = False)
        """

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = "SMPL_{}.{ext}".format(gender.upper(), ext="pkl")
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), "Path {} does not exist!".format(smpl_path)

            with open(smpl_path, "rb") as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM:
            print(
                f"WARNING: You are using a {self.name()} model, with only"
                f" {shapedirs.shape[-1]} shape coefficients.\n"
                f"num_betas={num_betas}, shapedirs.shape={shapedirs.shape}, "
                f"self.SHAPE_SPACE_DIM={self.SHAPE_SPACE_DIM}"
            )
            num_betas = min(num_betas, shapedirs.shape[-1])
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age == "kid":
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(
                v_template_smil - data_struct.v_template, axis=2
            )
            shapedirs = np.concatenate(
                (shapedirs[:, :, :num_betas], v_template_diff), axis=2
            )
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS["smplh"]

        self.vertex_ids = vertex_ids

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs
        )

        self.faces = data_struct.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                "betas", nn.Parameter(default_betas, requires_grad=True)
            )

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient, requires_grad=True)
            self.register_parameter("global_orient", global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype
                )
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                "body_pose", nn.Parameter(default_body_pose, requires_grad=True)
            )

        if create_transl:
            if transl is None:
                default_transl = torch.zeros(
                    [batch_size, 3], dtype=dtype, requires_grad=True
                )
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                "transl", nn.Parameter(default_transl, requires_grad=True)
            )

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer("v_template", v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer("lbs_weights", lbs_weights)

        if return_kpt:
            self.vertex_keypoint_regressor = VerticeKeypointRegressor(
                J_regressor=j_regressor,
                vertex_ids=vertex_ids,
                J_regressor_kpt=J_regressor_kpt,
            )
        else:
            self.vertex_keypoint_regressor = None

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return "SMPL"

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        raise NotImplementedError("should be shape[1]?")
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f"Gender: {self.gender.upper()}",
            f"Number of joints: {self.J_regressor.shape[0]}",
            f"Betas: {self.num_betas}",
        ]
        return "\n".join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLOutput:
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None and hasattr(self, "transl"):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        if full_pose.shape[0] != batch_size:
            num_repeats = int(batch_size / full_pose.shape[0])
            full_pose = full_pose.expand(num_repeats, -1).contiguous()

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        joints = self.vertex_joint_selector(vertices, joints)
        if self.vertex_keypoint_regressor is not None:
            joints = self.vertex_keypoint_regressor(vertices, joints)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            # NOTE(YL 09/24):: this in-place operation will
            # break the computation graph sometimes?
            # joints += transl.unsqueeze(dim=1)
            # vertices += transl.unsqueeze(dim=1)
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


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
