#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/20/2024
#
# Distributed under terms of the MIT license.

""" """

import os
import os.path as osp

import einops
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .smpl import create as fetal_smpl_create
from .utils import (
    _compute_kpt_loss,
    _compute_segm_loss,
    chamfer_distance,
    get_general_loss_func,
    resize_smpl_model,
    set_smil_model_to_fetal_smpl,
)
from .vertex_keypoint_regressor import _kpt_name_list


def main_init_smpl_beta_from_unposed(
    exp_result_dir,
    subj_name,
    smil_data_path,
    num_betas=10,
    fetal_smpl_data_dict_path=None,
    scale_body_size=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    exp_subj_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
    unposed_verts_list = np.load(
        osp.join(exp_subj_dir, "init_unposed", "segm_vertex_seq.npy"), allow_pickle=True
    )
    unposed_verts_list = np.concatenate(unposed_verts_list, axis=0)  # (n_total, 3)
    unposed_verts_list = np.array(unposed_verts_list, dtype=np.float32)
    unposed_verts_list = torch.tensor(unposed_verts_list, device=device)

    # add symmetric verts
    unposed_verts_symm_list = unposed_verts_list.clone()
    unposed_verts_symm_list[..., 0] = -unposed_verts_symm_list[..., 0]
    unposed_verts_list = torch.cat([unposed_verts_list, unposed_verts_symm_list], dim=0)

    # smil model
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        ext="npz",
        create_body_pose=True,
        create_global_orient=True,
        create_transl=True,
    )
    if fetal_smpl_data_dict_path is not None:
        set_smil_model_to_fetal_smpl(model, fetal_smpl_data_dict_path)

    if scale_body_size != 1:
        resize_smpl_model(model, scale_body_size)

    model = model.to(device)

    # shape (beta)
    beta = torch.zeros(
        [1, num_betas], dtype=torch.float32, requires_grad=True, device=device
    )

    # logging
    log_interval = 5

    write_file = osp.join(exp_result_dir, "tb_log", "init_smil_beta_unposed", subj_name)
    os.makedirs(write_file, exist_ok=True)
    writer = SummaryWriter(write_file)

    # scale dist
    scale_dist = 100

    # regularization
    coeff_reg_beta = 0.5

    # surface distance loss
    f_loss_smpl_to_segm = get_general_loss_func("geman_mcclure", device)
    f_loss_segm_to_smpl = get_general_loss_func("geman_mcclure", device)

    # only shape is optimized
    lr = 0.05
    num_step = 100
    optim = torch.optim.Adam([beta], lr=lr)

    beta_history = []
    shape_history = []

    for step in range(num_step):
        # smpl model produce vertices and joints
        # in 'xy' indexing
        output = model(betas=beta, return_verts=True)
        vertices = output.vertices[0]  # (6890, 3)

        reg_beta_per_dim = beta**2
        reg_beta = coeff_reg_beta * einops.reduce(reg_beta_per_dim, "t n -> ()", "mean")

        dist_smpl_to_segm = chamfer_distance(
            vertices * scale_dist, unposed_verts_list * scale_dist
        )
        loss_smpl_to_segm = f_loss_smpl_to_segm(dist_smpl_to_segm.view(-1, 1)).mean()

        dist_segm_to_smpl = chamfer_distance(
            unposed_verts_list * scale_dist, vertices * scale_dist
        )
        loss_segm_to_smpl = f_loss_segm_to_smpl(dist_segm_to_smpl.view(-1, 1)).mean()

        loss = loss_smpl_to_segm + loss_segm_to_smpl + reg_beta

        if step % log_interval == 0 or step == num_step - 1:
            beta_history.append(np.array(beta[0].detach().cpu().numpy()))
            shape_history.append(np.array(vertices.detach().cpu().numpy()))

            writer.add_scalar("loss_smpl_to_segm", loss_smpl_to_segm, step)
            writer.add_scalar("loss_segm_to_smpl", loss_segm_to_smpl, step)
            writer.add_scalar("loss_reg_beta", reg_beta, step)

            print(
                f"step: {step}, loss: {float(loss.item()):.4f}, "
                f"loss_smpl_to_segm: {float(loss_smpl_to_segm.item()):.4f}, "
                f"loss_segm_to_smpl: {float(loss_segm_to_smpl.item()):.4f}, "
                f"reg_beta: {float(reg_beta.item()):.4f}"
            )

        optim.zero_grad()
        loss.backward()
        optim.step()

    # save beta history
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, "init_unposed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "beta_his"), np.array(beta_history))
    np.save(osp.join(save_dir, "shape_his"), np.array(shape_history))

    model_faces = torch.tensor(
        np.array(model.faces, dtype=np.int32), dtype=torch.long, device=device
    )
    np.save(osp.join(save_dir, "faces.npy"), model_faces.detach().cpu().numpy())


def edge_elastic_regularization(vertices, vertices_init, faces, scale_dist: float = 1):
    """Compute the change in edge length and direction. Return L2 norm
    of the diff edge vector.

    Args:
        vertices (torch.tensor): (b, n, 3)
        vertices_init (torch.tensor): (b, n, 3)
        faces (torch.tensor): (m, 3)
        scale_dist (float, optional): Scale factor for distance to avoid small values.

    Returns:
        torch.tensor: (b, m, 3)
    """

    # vertices are measured in unit of meter, scale to avoid small values
    vertices = vertices * scale_dist
    vertices_init = vertices_init.detach() * scale_dist

    def _get_edges(v, f):
        v0 = v[:, f[:, 0], :]
        v1 = v[:, f[:, 1], :]
        v2 = v[:, f[:, 2], :]

        e0 = v1 - v0
        e1 = v2 - v0
        e2 = v2 - v1

        return e0, e1, e2

    e0, e1, e2 = _get_edges(vertices, faces)
    e0_init, e1_init, e2_init = _get_edges(vertices_init, faces)

    l2_norm_diff = torch.stack(
        [
            torch.norm(e0 - e0_init, dim=-1),
            torch.norm(e1 - e1_init, dim=-1),
            torch.norm(e2 - e2_init, dim=-1),
        ],
        dim=-1,
    )

    return l2_norm_diff


def main_pose_blend_shape_J_regressor_kpt(
    exp_data_split_dir, exp_result_dir, step_idx, subj_name_list, smil_data_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _to_pt_float_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

    # read data
    segm_vertex_seq_list = []  # list of list of [V, 3]
    kpt_seq_list = []  # list of [T, N_kpt, 3]
    transl_seq_list = []  # list of [T, 3]
    global_orient_seq_list = []  # list of [T, 3]
    body_pose_seq_list = []  # list of [T, 69]
    shape_seq_list = []  # list of [T, 6890, 3]
    vertices_init_list = []  # list of [T, 6890, 3]
    total_T = 0

    for subj_name in tqdm(subj_name_list):
        exp_subj_data_dir = osp.join(exp_data_split_dir, subj_name)

        # segmentation vertex to be aligned
        segm_vertex_seq = np.load(
            osp.join(exp_subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
        )
        segm_vertex_seq = [_to_pt_float_tensor(v) for v in segm_vertex_seq]
        _T = len(segm_vertex_seq)
        segm_vertex_seq_list.append(segm_vertex_seq)

        # keypoints to be aligned
        kpt_seq = np.load(osp.join(exp_subj_data_dir, "kpt_seq.npy"))
        kpt_seq = _to_pt_float_tensor(kpt_seq)
        assert len(kpt_seq) == _T
        kpt_seq_list.append(kpt_seq)

        # the result of current step alignment.
        exp_subj_result_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
        posed_dir = osp.join(exp_subj_result_dir, f"{step_idx}_posed")

        tr_seq = np.load(osp.join(posed_dir, "transl_seq_his.npy"))[-1]
        tr_seq = _to_pt_float_tensor(tr_seq)
        assert len(tr_seq) == _T
        transl_seq_list.append(tr_seq)

        go_seq = np.load(osp.join(posed_dir, "global_orient_seq_his.npy"))[-1]
        go_seq = _to_pt_float_tensor(go_seq)
        assert len(go_seq) == _T
        global_orient_seq_list.append(go_seq)

        bp_seq = np.load(osp.join(posed_dir, "body_pose_seq_his.npy"))[-1]
        bp_seq = _to_pt_float_tensor(bp_seq)
        assert len(bp_seq) == _T
        body_pose_seq_list.append(bp_seq)

        # shape from last step
        last_unposed_fname = (
            "init_unposed" if step_idx == 1 else f"{step_idx - 1}_unposed"
        )
        exp_subj_unposed_dir = osp.join(exp_subj_result_dir, last_unposed_fname)
        shape = np.load(osp.join(exp_subj_unposed_dir, "shape_his.npy"))[-1]
        shape = _to_pt_float_tensor(shape)
        assert shape.shape == (6890, 3), f"shape: {shape.shape}"
        shape_repeat = shape.repeat(_T, 1, 1)
        shape_seq_list.append(shape_repeat)

        # smil model
        model = fetal_smpl_create(
            smil_data_path,
            model_type="smpl",
            gender="infant",
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
            batch_size=_T,
            v_template=shape.detach().cpu().numpy(),
            return_kpt=True,
        )
        model.to(device)

        with torch.no_grad():
            # get init vertices from this subject's time series
            output = model(
                betas=torch.zeros([1, 10], dtype=torch.float32, device=device),
                body_pose=bp_seq,
                transl=tr_seq,
                global_orient=go_seq,
                return_verts=True,
            )
            vertices = output.vertices
            vertices_init_list.append(vertices.detach())

        # remove model from memory
        del model

        total_T += _T

    segm_vertex_seq_flat = []
    for v_seq in segm_vertex_seq_list:
        segm_vertex_seq_flat.extend(v_seq)
    kpt_seq_flat = torch.cat(kpt_seq_list, dim=0)
    transl_seq_flat = torch.cat(transl_seq_list, dim=0)
    global_orient_seq_flat = torch.cat(global_orient_seq_list, dim=0)
    body_pose_seq_flat = torch.cat(body_pose_seq_list, dim=0)
    shape_seq_flat = torch.cat(shape_seq_list, dim=0)
    vertices_init_flat = torch.cat(vertices_init_list, dim=0)

    # smil model
    # if v_template.shape[0] is not 1, the batch size should
    # always match v_template.shape[0] (subj num * T_sample)
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="npz",
        batch_size=total_T,
        v_template=shape_seq_flat.detach().cpu().numpy(),
        return_kpt=True,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
    )

    # make posedir optimizable
    posedirs_init = torch.tensor(model.posedirs.detach().cpu().numpy())
    del model.posedirs
    model.register_parameter("posedirs", torch.nn.Parameter(posedirs_init))

    # make J_regressor_kpt optimizable
    J_regressor_kpt_init = torch.tensor(
        model.vertex_keypoint_regressor.J_regressor_kpt.data
    )
    del model.vertex_keypoint_regressor.J_regressor_kpt
    model.vertex_keypoint_regressor.register_parameter(
        "J_regressor_kpt", torch.nn.Parameter(J_regressor_kpt_init)
    )

    model.to(device)

    # smil model for logging (inference)
    n_log_frames = 16
    idx_log_frames = np.linspace(0, total_T - 1, n_log_frames, dtype=int)
    idx_log_frames = torch.tensor(idx_log_frames, device=device)
    shape_seq_flat_for_log = shape_seq_flat[idx_log_frames]
    model_inf = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="npz",
        batch_size=n_log_frames,
        v_template=shape_seq_flat_for_log.detach().cpu().numpy(),
        return_kpt=True,
    )
    model_inf.to(device)

    # optimizer and schedule
    lr = 0.0003
    num_step = 50
    optim = torch.optim.Adam(
        [model.posedirs, model.vertex_keypoint_regressor.J_regressor_kpt], lr=lr
    )

    # scale distance: the distance is measured in unit of meter
    # this makes loss value very small.
    scale_dist = 100

    # robust loss
    f_loss_kpt = get_general_loss_func("geman_mcclure", device)
    f_loss_smpl_to_segm = get_general_loss_func("geman_mcclure", device)
    f_loss_segm_to_smpl = get_general_loss_func("geman_mcclure", device)

    # weight for the loss
    coeff_segm_chamfer = 1
    coeff_reg_posedirs = 0.01
    coeff_reg_J_regressor_kpt = 0.01
    coeff_reg_edge_elastic = 100

    # logging
    log_interval = 4
    log_his_interval = 4

    # log tensorboard
    log_dir = osp.join(
        exp_result_dir, "tb_log", f"{step_idx}_pose_blend_shape_J_regressor_kpt"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model_faces = torch.tensor(
        np.array(model.faces, dtype=np.int32), dtype=torch.long, device=device
    )

    pose_blend_shape_his = []  # [N_history, 69, 6890, 3]
    J_regressor_kpt_his = []  # [N_history, N_kpt, 6890]
    pred_kpt_list_his = []  # [N_history, N_log_frame, N_kpt, 3]
    pred_shape_list_his = []  # [N_history, N_log_frame, 6890, 3]
    batch_size = min(2**10, total_T)  # total for all singleton is 7700
    for step in range(num_step):
        # random sample a batch
        idx = torch.randperm(total_T)[:batch_size]
        bp = body_pose_seq_flat[idx]
        tr = transl_seq_flat[idx]
        go = global_orient_seq_flat[idx]
        kpt = kpt_seq_flat[idx]
        segm_v = [segm_vertex_seq_flat[i] for i in idx]
        v_init = vertices_init_flat[idx]
        shape = shape_seq_flat[idx]

        # re-set batch size and v_template in model
        with torch.no_grad():
            model.batch_size = len(idx)
            model.v_template.data = shape.detach()

        # smpl model produce vertices and joints
        # in 'xy' indexing
        output = model(
            betas=torch.zeros([1, 10], dtype=torch.float32, device=device),
            body_pose=bp,
            transl=tr,
            global_orient=go,
            return_verts=True,
        )
        joints = output.joints
        keypoints = joints[:, -len(_kpt_name_list) :, :]
        vertices = output.vertices

        # edge elastic regularization
        l2_norm_diff = edge_elastic_regularization(
            vertices, v_init, model_faces, scale_dist=scale_dist
        )  # (b, m, 3)
        reg_edge_elastic = coeff_reg_edge_elastic * l2_norm_diff.mean()

        # compute joint regression loss on joints in fetal
        loss_kpt = _compute_kpt_loss(keypoints, kpt, scale_dist, f_loss_kpt)

        # chamfer distance (l2, for now k=1)
        loss_segm_to_smpl, loss_smpl_to_segm = _compute_segm_loss(
            vertices,
            segm_v,
            scale_dist,
            f_loss_smpl_to_segm,
            f_loss_segm_to_smpl,
            coeff_segm_chamfer,
        )

        # regularize the posedirs to be sparse
        reg_posedirs = coeff_reg_posedirs * torch.norm(model.posedirs, p=1)

        # regularize the J_regressor_kpt to be close to init value
        J_regressor_kpt = model.vertex_keypoint_regressor.J_regressor_kpt
        J_regressor_kpt_init = model.vertex_keypoint_regressor.J_regressor_kpt_init
        reg_J_regressor_kpt = coeff_reg_J_regressor_kpt * torch.norm(
            J_regressor_kpt - J_regressor_kpt_init, p=2
        )

        loss = (
            loss_kpt
            + loss_smpl_to_segm
            + loss_segm_to_smpl
            + reg_posedirs
            + reg_J_regressor_kpt
            + reg_edge_elastic
        )

        if step % log_interval == 0 or step == num_step - 1:
            # log loss
            writer.add_scalar("loss/loss_kpt", loss_kpt.item(), step)
            writer.add_scalar("loss/loss_smpl_to_segm", loss_smpl_to_segm.item(), step)
            writer.add_scalar("loss/loss_segm_to_smpl", loss_segm_to_smpl.item(), step)
            writer.add_scalar("loss/reg_posedirs", reg_posedirs.item(), step)
            writer.add_scalar(
                "loss/reg_J_regressor_kpt", reg_J_regressor_kpt.item(), step
            )
            writer.add_scalar("loss/reg_edge_elastic", reg_edge_elastic.item(), step)
            writer.add_scalar("loss/all", loss.item(), step)

            print(
                f"step: {step}, loss: {loss.item():.4f}, "
                f"loss_kpt: {loss_kpt.item():.4f}, "
                f"loss_smpl_to_segm: {loss_smpl_to_segm.item():.4f}, "
                f"loss_segm_to_smpl: {loss_segm_to_smpl.item():.4f}, "
                f"reg_posedirs: {reg_posedirs.item():.4f}, "
                f"reg_J_regressor_kpt: {reg_J_regressor_kpt.item():.4f}, "
                f"reg_edge_elastic: {reg_edge_elastic.item():.4f}"
            )

        if step % log_his_interval == 0 or step == num_step - 1:
            pose_blend_shape_his.append(np.array(model.posedirs.detach().cpu().numpy()))
            J_regressor_kpt_np = (
                model.vertex_keypoint_regressor.J_regressor_kpt.detach().cpu().numpy()
            )
            J_regressor_kpt_his.append(np.array(J_regressor_kpt_np))

            # log the predicted keypoints and shape
            with torch.no_grad():
                output = model_inf(
                    betas=torch.zeros(
                        [n_log_frames, 10], dtype=torch.float32, device=device
                    ),
                    body_pose=body_pose_seq_flat[idx_log_frames],
                    transl=transl_seq_flat[idx_log_frames],
                    global_orient=global_orient_seq_flat[idx_log_frames],
                    return_verts=True,
                    return_kpt=True,
                )
                pred_kpt_list = output.joints[:, -len(_kpt_name_list) :, :]
                pred_shape_list = output.vertices

            pred_kpt_list_his.append(np.array(pred_kpt_list.detach().cpu().numpy()))
            pred_shape_list_his.append(np.array(pred_shape_list.detach().cpu().numpy()))

        loss = loss.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            # set non support values to zero
            J_regressor_kpt_support = (
                model.vertex_keypoint_regressor.J_regressor_kpt_support
            )
            J_regressor_kpt.data = J_regressor_kpt.data * J_regressor_kpt_support

            # for J regressor: non-negative constraint
            # and normalize to sum to 1 for gradient decent.
            J_regressor_kpt.data = torch.clamp(J_regressor_kpt, min=0)
            J_regressor_kpt.data = J_regressor_kpt / J_regressor_kpt.sum(
                dim=-1, keepdim=True
            )

    # save data
    save_dir = osp.join(
        exp_result_dir,
        "population",
        f"{step_idx}_pose_blend_shape_J_regressor_kpt",
    )
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        osp.join(save_dir, "pose_blend_shape_his"),
        pose_blend_shape_his,
    )
    np.save(
        osp.join(save_dir, "J_regressor_kpt_his"),
        J_regressor_kpt_his,
    )

    kpt_seq_flat_for_log = kpt_seq_flat[idx_log_frames].cpu().numpy()
    segm_vertex_seq_flat_for_log = [
        segm_vertex_seq_flat[i].cpu().numpy() for i in idx_log_frames
    ]
    segm_vertex_seq_flat_for_log = np.array(segm_vertex_seq_flat_for_log, dtype=object)
    np.save(osp.join(save_dir, "kpt_seq"), kpt_seq_flat_for_log)
    np.save(osp.join(save_dir, "segm_vertex_seq"), segm_vertex_seq_flat_for_log)

    pred_kpt_list_his = np.stack(pred_kpt_list_his, axis=0)
    pred_shape_list_his = np.stack(pred_shape_list_his, axis=0)
    np.save(osp.join(save_dir, "pred_kpt_list_his"), pred_kpt_list_his)
    np.save(osp.join(save_dir, "pred_shape_list_his"), pred_shape_list_his)


def main_learn_subj_spec_shape_from_unposed(
    exp_result_dir, step_idx, subj_name, smil_data_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read unposed kpt and vertex data
    exp_subj_result_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
    exp_subj_unposed_dir = osp.join(exp_subj_result_dir, f"{step_idx}_unposed")
    unposed_verts_list = np.load(
        osp.join(exp_subj_unposed_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )
    unposed_verts_list = np.concatenate(unposed_verts_list, axis=0)  # (n_total, 3)
    unposed_verts_list = torch.tensor(unposed_verts_list, device=device)

    # add symmetric verts
    unposed_verts_symm_list = unposed_verts_list.clone()
    unposed_verts_symm_list[..., 0] = -unposed_verts_symm_list[..., 0]
    unposed_verts_list = torch.cat([unposed_verts_list, unposed_verts_symm_list], dim=0)

    # read shape from last step
    last_unposed_fname = "init_unposed" if step_idx == 1 else f"{step_idx - 1}_unposed"
    last_unposed_dir = osp.join(exp_subj_result_dir, last_unposed_fname)
    subj_spec_shape_init = np.load(osp.join(last_unposed_dir, "shape_his.npy"))[-1]

    # smil model
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="npz",
        create_body_pose=True,
        create_global_orient=True,
        create_transl=True,
    )

    # make v_template optimizable and init with subj spec shape
    subj_spec_shape_init = torch.tensor(
        subj_spec_shape_init, dtype=torch.float32, device=device
    )
    del model.v_template
    model.register_parameter("v_template", torch.nn.Parameter(subj_spec_shape_init))

    model = model.to(device)

    vertices_init = None  # [NxT, 6890, 3]
    model_faces = torch.tensor(
        np.array(model.faces, dtype=np.int32), dtype=torch.long, device=device
    )

    # logging
    log_interval = 10
    shape_history = []

    write_file = osp.join(
        exp_result_dir, "tb_log", f"{step_idx}_subj_spec_shape", subj_name
    )
    os.makedirs(write_file, exist_ok=True)
    writer = SummaryWriter(write_file)

    # scale dist
    scale_dist = 100

    # surface alignment loss
    f_loss_smpl_to_segm = get_general_loss_func("geman_mcclure", device)
    f_loss_segm_to_smpl = get_general_loss_func("geman_mcclure", device)

    # elastic regularization
    coeff_reg_edge_elastic = 50

    # only shape is optimized
    lr = 0.005
    num_step = 25
    optim = torch.optim.Adam([model.v_template], lr=lr)

    for step in range(num_step):
        # smpl model produce vertices and joints
        # in 'xy' indexing
        output = model(
            betas=torch.zeros([1, 10], dtype=torch.float32, device=device),
            return_verts=True,
        )
        vertices = output.vertices[0]  # (6890, 3)

        # chamfer distance (l2, for now k=1)
        dist_smpl_to_segm = chamfer_distance(
            vertices * scale_dist, unposed_verts_list * scale_dist
        )
        loss_smpl_to_segm = f_loss_smpl_to_segm(dist_smpl_to_segm.view(-1, 1)).mean()

        dist_segm_to_smpl = chamfer_distance(
            unposed_verts_list * scale_dist, vertices * scale_dist
        )
        loss_segm_to_smpl = f_loss_segm_to_smpl(dist_segm_to_smpl.view(-1, 1)).mean()

        # elastic regularization
        if vertices_init is None:
            vertices_init = vertices.detach()[None]  # (1, 6890, 3)
        l2_norm_diff = edge_elastic_regularization(
            vertices[None], vertices_init, model_faces, scale_dist=scale_dist
        )  # (1, m, 3)
        reg_edge_elastic = coeff_reg_edge_elastic * l2_norm_diff.mean()

        loss = loss_smpl_to_segm + loss_segm_to_smpl + reg_edge_elastic

        if step % log_interval == 0 or step == num_step - 1:
            shape_history.append(np.array(vertices.detach().cpu().numpy()))

            writer.add_scalar("loss_smpl_to_segm", loss_smpl_to_segm, step)
            writer.add_scalar("loss_segm_to_smpl", loss_segm_to_smpl, step)
            writer.add_scalar("reg_edge_elastic", reg_edge_elastic, step)

            print(
                f"step: {step}, loss: {float(loss.item()):.4f}, "
                f"loss_smpl_to_segm: {float(loss_smpl_to_segm.item()):.4f}, "
                f"loss_segm_to_smpl: {float(loss_segm_to_smpl.item()):.4f}, "
                f"reg_edge_elastic: {float(reg_edge_elastic.item()):.4f}"
            )

        optim.zero_grad()
        loss.backward()
        optim.step()

    # save beta history
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, f"{step_idx}_unposed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "shape_his.npy"), np.array(shape_history))
    # save model face
    np.save(osp.join(save_dir, "faces.npy"), model_faces.detach().cpu().numpy())


def main_shape_blend_shape_and_save_model(exp_result_dir, step_idx, subj_name_list):
    """Apply PCA to subj specific shape to get shape blend shape.
    Read pose blend shape and J_regressor_kpt. Then save model.
    """

    from sklearn.decomposition import PCA

    # read all subject's shape data
    shape_list = []
    for subj_name in subj_name_list:
        exp_subj_result_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
        shape_his_path = osp.join(
            exp_subj_result_dir, f"{step_idx}_unposed", "shape_his.npy"
        )
        shape = np.load(shape_his_path)[-1]
        shape_list.append(shape)
    shape_list = np.stack(shape_list, axis=0)  # (N, 6890, 3)
    n_vertex = shape_list.shape[1]
    print("shape_list.shape: {}".format(shape_list.shape))

    # mean shape
    mean_shape = shape_list.mean(axis=0)  # (6890, 3)
    c_shape_list = shape_list - mean_shape[None]
    c_shape_flat = einops.rearrange(c_shape_list, "n v c -> n (v c)")

    # apply PCA to get shape blend shape
    n_components = 32
    pca = PCA(n_components=n_components)
    pca.fit(c_shape_flat)
    print(
        "pca.explained_variance_ratio_: {}".format(
            [f"{x:.4f}" for x in pca.explained_variance_ratio_]
        )
    )

    # get shape blend shape
    shape_blend_shape = pca.components_.reshape(n_components, n_vertex, 3)
    shape_blend_shape = einops.rearrange(
        shape_blend_shape, "c v d -> v d c"
    )  # smpl format
    print("shape_blend_shape.shape: {}".format(shape_blend_shape.shape))

    # read pose blend shape and J_regressor_kpt
    population_dir = osp.join(
        exp_result_dir,
        "population",
        f"{step_idx}_pose_blend_shape_J_regressor_kpt",
    )
    pose_blend_shape_history = np.load(
        osp.join(population_dir, "pose_blend_shape_his.npy")
    )
    pose_blend_shape = pose_blend_shape_history[-1]  # (207, 6890x3)
    print("pose_blend_shape.shape: {}".format(pose_blend_shape.shape))
    J_regressor_kpt_history = np.load(
        osp.join(population_dir, "J_regressor_kpt_his.npy")
    )
    J_regressor_kpt = J_regressor_kpt_history[-1]  # (N_kpt, 6890)
    print("J_regressor_kpt.shape: {}".format(J_regressor_kpt.shape))

    # save data
    data_dict = {
        "mean_shape": mean_shape,
        "shape_blend_shape": shape_blend_shape,
        "pose_blend_shape": pose_blend_shape,
        "J_regressor_kpt": J_regressor_kpt,
    }
    save_file = osp.join(exp_result_dir, "model", f"step_{step_idx}.npy")
    os.makedirs(osp.dirname(save_file), exist_ok=True)
    np.save(save_file, data_dict)
