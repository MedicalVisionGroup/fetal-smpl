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

import numpy as np
import torch
from einops import reduce
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .smpl import create as fetal_smpl_create
from .evaluation import evaluate_alignment_to_posed_segm_kpt
from .lbs_unpose import unpose_keypoint_vertice
from .utils import (
    _compute_kpt_loss,
    _compute_segm_loss,
    get_general_loss_func,
    resize_smpl_model,
    set_smil_model_to_fetal_smpl,
)
from .vertex_keypoint_regressor import _kpt_name_list, kpt2smpl_joint_parent_idx_map


def _align_smpl_to_posed(
    model,
    beta,
    body_pose,
    global_orient,
    transl,
    kpt_seq,
    segm_vertex_seq,
    prior_mean,
    prior_prec_decomposed,
    scale_dist,
    coeff_reg_beta,
    coeff_reg_body_pose,
    coeff_smooth_global_orient,
    coeff_smooth_transl,
    coeff_smooth_body_pose,
    coeff_segm_chamfer,
    num_step_global_pose,
    num_step_wo_segm,
    num_step_w_segm,
    log_interval,
    device,
    writer,
    optim_global_pose,
    optim_wo_segm,
    optim_w_segm,
):
    # logging the history during optimization
    body_pose_seq_his = []
    beta_his = []
    transl_seq_his = []
    global_orient_seq_his = []
    kpt_pred_seq_his = []

    def _compute_reg():
        # regularization: beta and body pose
        reg_beta_per_dim = beta**2
        reg_beta = coeff_reg_beta * reduce(reg_beta_per_dim, "t n -> ()", "mean")

        reg_body_pose_projected = torch.einsum(
            "t i, i j -> t j", body_pose - prior_mean[None], prior_prec_decomposed
        )
        reg_body_pose_per_joint = reg_body_pose_projected**2
        reg_body_pose = coeff_reg_body_pose * reduce(
            reg_body_pose_per_joint, "t n -> ()", "mean"
        )

        # regularize temporal smoothness in
        # global_orient, transl, and body_pose
        def _temp_smooth_reg_l2(v, _scale):
            diff = v[1:] - v[:-1]
            diff = diff * _scale
            # t n 3 -> () or t 3 -> ()
            loss = (diff**2).sum(dim=-1)
            loss = loss.mean()
            return loss

        reg_smooth_global_orient = coeff_smooth_global_orient * _temp_smooth_reg_l2(
            global_orient, scale_dist
        )
        reg_smooth_transl = coeff_smooth_transl * _temp_smooth_reg_l2(
            transl, scale_dist
        )
        reg_smooth_body_pose = coeff_smooth_body_pose * _temp_smooth_reg_l2(
            body_pose, scale_dist
        )

        return (
            reg_beta,
            reg_body_pose,
            reg_smooth_global_orient,
            reg_smooth_transl,
            reg_smooth_body_pose,
        )

    def _log_kpt_reg(
        loss_kpt,
        reg_beta,
        reg_body_pose,
        reg_smooth_global_orient,
        reg_smooth_transl,
        reg_smooth_body_pose,
    ):
        writer.add_scalar("loss/loss_kpt", loss_kpt.item(), step)
        writer.add_scalar("loss/reg_beta", reg_beta.item(), step)
        writer.add_scalar("loss/reg_body_pose", reg_body_pose.item(), step)
        writer.add_scalar(
            "loss/reg_smooth_global_orient", reg_smooth_global_orient, step
        )
        writer.add_scalar("loss/reg_smooth_transl", reg_smooth_transl, step)
        writer.add_scalar("loss/reg_smooth_body_pose", reg_smooth_body_pose, step)

        print(
            f"loss_kpt: {float(loss_kpt.item()):.4f}, "
            f"reg_beta: {float(reg_beta.item()):.4f}, "
            f"reg_body_pose: {float(reg_body_pose.item()):.4f}, "
            f"reg_smooth_global_orient: {float(reg_smooth_global_orient):.4f}, "
            f"reg_smooth_transl: {float(reg_smooth_transl):.4f}, "
            f"reg_smooth_body_pose: {float(reg_smooth_body_pose):.4f}"
        )

    total_step = num_step_global_pose + num_step_wo_segm + num_step_w_segm

    # optimization without segmentation
    # noise robust loss is only applied in final stage with segmentation
    f_loss_kpt = get_general_loss_func("l2", device)
    for step in range(num_step_global_pose):
        # smpl model produce vertices and joints
        # in 'xy' indexing
        output = model(
            betas=beta,
            body_pose=body_pose,
            transl=transl,
            global_orient=global_orient,
            return_verts=False,
        )
        joints = output.joints
        keypoints = joints[:, -len(_kpt_name_list) :, :]

        loss_kpt = _compute_kpt_loss(keypoints, kpt_seq, scale_dist, f_loss_kpt)
        (
            reg_beta,
            reg_body_pose,
            reg_smooth_global_orient,
            reg_smooth_transl,
            reg_smooth_body_pose,
        ) = _compute_reg()

        loss = loss_kpt + reg_smooth_global_orient + reg_smooth_transl

        if step % log_interval == 0:
            writer.add_scalar("loss/all", loss.item(), step)
            print(f"step: {step}, loss: {float(loss.item()):.4f}, ")
            _log_kpt_reg(
                loss_kpt,
                reg_beta,
                reg_body_pose,
                reg_smooth_global_orient,
                reg_smooth_transl,
                reg_smooth_body_pose,
            )

            # log the body_pose, betas, transl, global_orient history
            body_pose_seq_his.append(np.array(body_pose.detach().cpu().numpy()))
            beta_his.append(np.array(beta.detach().cpu().numpy()))
            transl_seq_his.append(np.array(transl.detach().cpu().numpy()))
            global_orient_seq_his.append(np.array(global_orient.detach().cpu().numpy()))
            kpt_pred_seq_his.append(np.array(keypoints.detach().cpu().numpy()))

        optim_global_pose.zero_grad()
        loss.backward()
        optim_global_pose.step()

    for step in range(num_step_global_pose, num_step_wo_segm + num_step_global_pose):
        # smpl model produce vertices and joints
        # in 'xy' indexing
        output = model(
            betas=beta,
            body_pose=body_pose,
            transl=transl,
            global_orient=global_orient,
            return_verts=False,
        )
        joints = output.joints
        keypoints = joints[:, -len(_kpt_name_list) :, :]

        loss_kpt = _compute_kpt_loss(keypoints, kpt_seq, scale_dist, f_loss_kpt)
        (
            reg_beta,
            reg_body_pose,
            reg_smooth_global_orient,
            reg_smooth_transl,
            reg_smooth_body_pose,
        ) = _compute_reg()

        loss = (
            loss_kpt
            + reg_beta
            + reg_body_pose
            + reg_smooth_global_orient
            + reg_smooth_transl
            + reg_smooth_body_pose
        )

        if step % log_interval == 0 or step == total_step - 1:
            writer.add_scalar("loss/all", loss.item(), step)
            print(f"step: {step}, loss: {float(loss.item()):.4f}, ")
            _log_kpt_reg(
                loss_kpt,
                reg_beta,
                reg_body_pose,
                reg_smooth_global_orient,
                reg_smooth_transl,
                reg_smooth_body_pose,
            )

            # log the body_pose, betas, transl, global_orient history
            body_pose_seq_his.append(np.array(body_pose.detach().cpu().numpy()))
            beta_his.append(np.array(beta.detach().cpu().numpy()))
            transl_seq_his.append(np.array(transl.detach().cpu().numpy()))
            global_orient_seq_his.append(np.array(global_orient.detach().cpu().numpy()))
            kpt_pred_seq_his.append(np.array(keypoints.detach().cpu().numpy()))

        optim_wo_segm.zero_grad()
        loss.backward()
        optim_wo_segm.step()

    # optimization with segmentation (noise robust loss)
    f_loss_kpt = get_general_loss_func("geman_mcclure", device)
    f_loss_smpl_to_segm = get_general_loss_func("geman_mcclure", device)
    f_loss_segm_to_smpl = get_general_loss_func("geman_mcclure", device)
    for step in range(num_step_wo_segm + num_step_global_pose, total_step):
        # smpl model produce vertices and joints
        # in 'xy' indexing
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

        loss_kpt = _compute_kpt_loss(keypoints, kpt_seq, scale_dist, f_loss_kpt)
        loss_segm_to_smpl, loss_smpl_to_segm = _compute_segm_loss(
            vertices,
            segm_vertex_seq,
            scale_dist,
            f_loss_smpl_to_segm,
            f_loss_segm_to_smpl,
            coeff_segm_chamfer,
        )
        (
            reg_beta,
            reg_body_pose,
            reg_smooth_global_orient,
            reg_smooth_transl,
            reg_smooth_body_pose,
        ) = _compute_reg()

        loss = (
            loss_kpt
            + reg_beta
            + reg_body_pose
            + reg_smooth_global_orient
            + reg_smooth_transl
            + reg_smooth_body_pose
            + loss_smpl_to_segm
            + loss_segm_to_smpl
        )

        if step % log_interval == 0 or step == total_step - 1:
            writer.add_scalar("loss/all", loss.item(), step)
            print(f"step: {step}, loss: {float(loss.item()):.4f}, ")
            _log_kpt_reg(
                loss_kpt,
                reg_beta,
                reg_body_pose,
                reg_smooth_global_orient,
                reg_smooth_transl,
                reg_smooth_body_pose,
            )

            writer.add_scalar("loss/loss_smpl_to_segm", loss_smpl_to_segm.item(), step)
            writer.add_scalar("loss/loss_segm_to_smpl", loss_segm_to_smpl.item(), step)
            print(
                f"loss_smpl_to_segm: {float(loss_smpl_to_segm.item()):.4f}, "
                f"loss_segm_to_smpl: {float(loss_segm_to_smpl.item()):.4f}, "
            )

            # log the body_pose, betas, transl, global_orient history
            body_pose_seq_his.append(np.array(body_pose.detach().cpu().numpy()))
            beta_his.append(np.array(beta.detach().cpu().numpy()))
            transl_seq_his.append(np.array(transl.detach().cpu().numpy()))
            global_orient_seq_his.append(np.array(global_orient.detach().cpu().numpy()))
            kpt_pred_seq_his.append(np.array(keypoints.detach().cpu().numpy()))

        optim_w_segm.zero_grad()
        loss.backward()
        optim_w_segm.step()

    return (
        body_pose_seq_his,
        beta_his,
        transl_seq_his,
        global_orient_seq_his,
        kpt_pred_seq_his,
    )


def main_init_align_smpl_to_posed_and_unpose(
    exp_data_split_dir,
    exp_result_dir,
    subj_name,
    smil_data_path,
    num_betas=10,
    fetal_smpl_data_dict_path=None,
    scale_body_size=1,
):
    # read data
    exp_subj_data_dir = osp.join(exp_data_split_dir, subj_name)
    kpt_seq = np.load(osp.join(exp_subj_data_dir, "kpt_seq.npy"))
    segm_vertex_seq = np.load(
        osp.join(exp_subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )

    # np to pt array, then move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kpt_seq = torch.tensor(kpt_seq, dtype=torch.float32, device=device)
    segm_vertex_seq = [
        torch.tensor(v, dtype=torch.float32, device=device) for v in segm_vertex_seq
    ]

    # smil model
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        ext="npz",
        return_kpt=True,
    )
    # set posedir to zero to distable pose blend shape
    model.posedirs.fill_(0)

    if fetal_smpl_data_dict_path is not None:
        set_smil_model_to_fetal_smpl(model, fetal_smpl_data_dict_path)

    if scale_body_size != 1:
        resize_smpl_model(model, scale_body_size)

    model = model.to(device)

    # smil pose prior: pose regularization using mahalanobis distance
    # the file actually contains the spectral decomposition of precision matrix
    prior_mean = np.load(osp.join(smil_data_path, "smil_pose_prior_converted_mean.npy"))
    prior_mean = torch.tensor(prior_mean, dtype=torch.float32, device=device)
    prior_prec_decomposed = np.load(
        osp.join(smil_data_path, "smil_pose_prior_converted_prec.npy")
    )
    prior_prec_decomposed = torch.tensor(
        prior_prec_decomposed, dtype=torch.float32, device=device
    )

    # pose (body_pose, global_orient, transl)
    # initialize pose with mean pose from smil prior
    # initialize transl with mean pelvis position
    T = kpt_seq.shape[0]
    body_pose = torch.tensor(
        torch.stack([prior_mean] * T, dim=0),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    pelvis = kpt_seq[:, _kpt_name_list.index("bladder")]
    pelvis_mean = pelvis.mean(dim=0)
    transl = torch.tensor(
        torch.stack([pelvis_mean] * T, dim=0),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    global_orient = torch.zeros(
        [T, 3], dtype=torch.float32, requires_grad=True, device=device
    )

    # shape (beta)
    beta = torch.zeros(
        [1, num_betas], dtype=torch.float32, requires_grad=True, device=device
    )

    # optimizer and schedule
    lr = 0.05
    num_step_global_pose = 150
    optim_global_pose = torch.optim.Adam([transl, global_orient], lr=lr)

    lr = 0.01
    num_step_wo_segm = 150
    optim_wo_segm = torch.optim.Adam([beta, body_pose, transl, global_orient], lr=lr)

    lr = 0.003
    num_step_w_segm = 100
    optim_w_segm = torch.optim.Adam([beta, body_pose, transl, global_orient], lr=lr)

    # scale distance: the distance is measured in unit of meter
    # this makes loss value very small.
    scale_dist = 100

    # loss weights
    coeff_segm_chamfer = 50
    coeff_reg_beta = 0.1
    coeff_reg_body_pose = 0.1

    coeff_smooth_global_orient = 0.001
    coeff_smooth_transl = 0.001
    coeff_smooth_body_pose = 0.0001

    # logging
    log_interval = 10

    # log tensorboard
    log_dir = osp.join(exp_result_dir, "tb_log", "init_align_smil", subj_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # optimization
    (
        body_pose_seq_his,
        beta_his,
        transl_seq_his,
        global_orient_seq_his,
        kpt_pred_seq_his,
    ) = _align_smpl_to_posed(
        model,
        beta,
        body_pose,
        global_orient,
        transl,
        kpt_seq,
        segm_vertex_seq,
        prior_mean,
        prior_prec_decomposed,
        scale_dist,
        coeff_reg_beta,
        coeff_reg_body_pose,
        coeff_smooth_global_orient,
        coeff_smooth_transl,
        coeff_smooth_body_pose,
        coeff_segm_chamfer,
        num_step_global_pose,
        num_step_wo_segm,
        num_step_w_segm,
        log_interval,
        device,
        writer,
        optim_global_pose,
        optim_wo_segm,
        optim_w_segm,
    )

    # save data
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, "init_posed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "transl_seq_his"), np.array(transl_seq_his))
    np.save(
        os.path.join(save_dir, "global_orient_seq_his"),
        np.array(global_orient_seq_his),
    )
    np.save(osp.join(save_dir, "body_pose_seq_his"), np.array(body_pose_seq_his))
    np.save(osp.join(save_dir, "beta_his"), np.array(beta_his))
    np.save(osp.join(save_dir, "kpt_pred_seq_his"), np.array(kpt_pred_seq_his))

    # evaluate goodness of alignment and save metric
    with torch.no_grad():
        evaluate_alignment_to_posed_segm_kpt(
            model,
            beta,
            body_pose,
            global_orient,
            transl,
            kpt_seq,
            segm_vertex_seq,
            osp.join(save_dir, "eval"),
            scale_dist,
        )

    # kpt and segm to be unposed
    kpt_seq = np.load(osp.join(exp_subj_data_dir, "kpt_seq.npy"))
    body_seqm_verts_ts = np.load(
        osp.join(exp_subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )

    kpt2smpl_joint_parent_idx_map_pt = torch.tensor(
        kpt2smpl_joint_parent_idx_map, dtype=torch.long, device=device
    )

    unposed_joints_list = []
    unposed_vertices_list = []
    for t in tqdm(range(T)):
        pose_seq = torch.from_numpy(kpt_seq[t][None]).to(device).float()
        body_segm_verts = (
            torch.from_numpy(body_seqm_verts_ts[t][None]).to(device).float()
        )

        with torch.no_grad():
            keypoints_unposed, vertices_unposed = unpose_keypoint_vertice(
                model,
                beta,
                body_pose[t][None],
                transl[t][None],
                global_orient[t][None],
                pose_seq,
                body_segm_verts,
                kpt2smpl_joint_parent_idx_map_pt,
            )
            joints = keypoints_unposed.detach().cpu().numpy()
            vertices = vertices_unposed.detach().cpu().numpy()

        unposed_joints_list.append(joints[0])
        unposed_vertices_list.append(vertices[0])

    # save data
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, "init_unposed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        osp.join(save_dir, "segm_vertex_seq"),
        np.array(unposed_vertices_list, dtype=object),
    )
    np.save(osp.join(save_dir, "keypoint_seq"), np.array(unposed_joints_list))


def main_align_subj_spec_shape_to_posed(
    exp_data_split_dir,
    exp_result_dir,
    step_idx,
    subj_name,
    smil_data_path,
    num_betas=10,
    fetal_smpl_data_dict_path=None,
    scale_body_size=1,
):
    # read data
    exp_subj_data_dir = osp.join(exp_data_split_dir, subj_name)
    kpt_seq = np.load(osp.join(exp_subj_data_dir, "kpt_seq.npy"))
    segm_vertex_seq = np.load(
        osp.join(exp_subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )

    # read subj spec shape for last step
    exp_subj_result_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
    last_unposed_fname = "init_unposed" if step_idx == 1 else f"{step_idx - 1}_unposed"
    last_subj_spec_shape_path = osp.join(
        exp_subj_result_dir, last_unposed_fname, "shape_his.npy"
    )
    subj_spec_shape = np.load(last_subj_spec_shape_path)[-1]

    # np to pt array, then move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kpt_seq = torch.tensor(kpt_seq, dtype=torch.float32, device=device)
    segm_vertex_seq = [
        torch.tensor(v, dtype=torch.float32, device=device) for v in segm_vertex_seq
    ]

    # load last step's j_regressor_kpt
    if step_idx == 1:
        J_regressor_kpt = None
    else:
        last_j_regressor_kpt_path = osp.join(
            exp_result_dir,
            "population",
            f"{step_idx - 1}_pose_blend_shape_J_regressor_kpt",
            "J_regressor_kpt_his.npy",
        )
        J_regressor_kpt = np.load(last_j_regressor_kpt_path)[-1]

    # smil model
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        ext="npz",
        v_template=subj_spec_shape,
        return_kpt=True,
        J_regressor_kpt=J_regressor_kpt,
    )

    # load last step's pose blend shape
    if step_idx == 1:
        model.posedirs.fill_(0)
    else:
        last_pose_blend_shape_path = osp.join(
            exp_result_dir,
            "population",
            f"{step_idx - 1}_pose_blend_shape_J_regressor_kpt",
            "pose_blend_shape_his.npy",
        )
        pose_blend_shape = np.load(last_pose_blend_shape_path)[-1]
        pose_blend_shape = torch.tensor(pose_blend_shape, dtype=torch.float32)
        del model.posedirs
        model.register_buffer("posedirs", pose_blend_shape)

    if fetal_smpl_data_dict_path is not None:
        set_smil_model_to_fetal_smpl(model, fetal_smpl_data_dict_path, subj_spec_shape)

    if scale_body_size != 1:
        resize_smpl_model(model, scale_body_size)

    model = model.to(device)

    # smil pose prior: pose regularization using mahalanobis distance
    # the file actually contains the spectral decomposition of precision matrix
    prior_mean = np.load(osp.join(smil_data_path, "smil_pose_prior_converted_mean.npy"))
    prior_mean = torch.tensor(prior_mean, dtype=torch.float32, device=device)
    prior_prec_decomposed = np.load(
        osp.join(smil_data_path, "smil_pose_prior_converted_prec.npy")
    )
    prior_prec_decomposed = torch.tensor(
        prior_prec_decomposed, dtype=torch.float32, device=device
    )

    # pose (body_pose, global_orient, transl)
    # initialize pose with mean pose from smil prior
    # initialize transl with mean pelvis position
    T = kpt_seq.shape[0]
    body_pose = torch.tensor(
        torch.stack([prior_mean] * T, dim=0),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    pelvis = kpt_seq[:, _kpt_name_list.index("bladder")]
    pelvis_mean = pelvis.mean(dim=0)
    transl = torch.tensor(
        torch.stack([pelvis_mean] * T, dim=0),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    global_orient = torch.zeros(
        [T, 3], dtype=torch.float32, requires_grad=True, device=device
    )

    # shape (beta)
    beta = torch.zeros(
        [1, num_betas], dtype=torch.float32, requires_grad=False, device=device
    )

    # optimizer and schedule
    lr = 0.05
    num_step_global_pose = 150
    optim_global_pose = torch.optim.Adam([transl, global_orient], lr=lr)

    lr = 0.01
    num_step_wo_segm = 150
    optim_wo_segm = torch.optim.Adam([beta, body_pose, transl, global_orient], lr=lr)

    lr = 0.003
    num_step_w_segm = 100
    optim_w_segm = torch.optim.Adam([body_pose, transl, global_orient], lr=lr)

    # scale distance: the distance is measured in unit of meter
    # this makes loss value very small.
    scale_dist = 100

    # loss weights
    coeff_segm_chamfer = 50
    coeff_reg_beta = 0.0
    coeff_reg_body_pose = 0.1

    coeff_smooth_global_orient = 0.001
    coeff_smooth_transl = 0.001
    coeff_smooth_body_pose = 0.0001

    # logging
    log_interval = 10

    # log tensorboard
    log_dir = osp.join(
        exp_result_dir, "tb_log", f"{step_idx}_align_subj_spec_shape", subj_name
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # optimization
    (
        body_pose_seq_his,
        beta_his,
        transl_seq_his,
        global_orient_seq_his,
        kpt_pred_seq_his,
    ) = _align_smpl_to_posed(
        model,
        beta,
        body_pose,
        global_orient,
        transl,
        kpt_seq,
        segm_vertex_seq,
        prior_mean,
        prior_prec_decomposed,
        scale_dist,
        coeff_reg_beta,
        coeff_reg_body_pose,
        coeff_smooth_global_orient,
        coeff_smooth_transl,
        coeff_smooth_body_pose,
        coeff_segm_chamfer,
        num_step_global_pose,
        num_step_wo_segm,
        num_step_w_segm,
        log_interval,
        device,
        writer,
        optim_global_pose,
        optim_wo_segm,
        optim_w_segm,
    )

    # save data
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, f"{step_idx}_posed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "transl_seq_his"), np.array(transl_seq_his))
    np.save(
        os.path.join(save_dir, "global_orient_seq_his"),
        np.array(global_orient_seq_his),
    )
    np.save(osp.join(save_dir, "body_pose_seq_his"), np.array(body_pose_seq_his))
    np.save(osp.join(save_dir, "kpt_pred_seq_his"), np.array(kpt_pred_seq_his))

    # evaluate goodness of alignment and save metric
    with torch.no_grad():
        evaluate_alignment_to_posed_segm_kpt(
            model,
            beta,
            body_pose,
            global_orient,
            transl,
            kpt_seq,
            segm_vertex_seq,
            osp.join(save_dir, "eval"),
            scale_dist,
        )


def main_unpose(
    exp_data_split_dir, exp_result_dir, step_idx, subj_name, smil_data_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read posed vertex and kpt data
    exp_subj_data_dir = osp.join(exp_data_split_dir, subj_name)
    kpt_seq = np.load(osp.join(exp_subj_data_dir, "kpt_seq.npy"))
    segm_vertex_seq = np.load(
        osp.join(exp_subj_data_dir, "segm_vertex_seq.npy"), allow_pickle=True
    )

    # read body_pose, transl, global_orient from posed alignment step
    exp_subj_result_dir = osp.join(exp_result_dir, "subj_spec", subj_name)
    posed_dir = osp.join(exp_subj_result_dir, f"{step_idx}_posed")
    body_pose_seq = np.load(osp.join(posed_dir, "body_pose_seq_his.npy"))[-1]
    transl_seq = np.load(osp.join(posed_dir, "transl_seq_his.npy"))[-1]
    global_orient_seq = np.load(osp.join(posed_dir, "global_orient_seq_his.npy"))[-1]

    # read shape from last step
    last_unposed_fname = "init_unposed" if step_idx == 1 else f"{step_idx - 1}_unposed"
    last_unposed_dir = osp.join(exp_subj_result_dir, last_unposed_fname)
    shape = np.load(osp.join(last_unposed_dir, "shape_his.npy"))[-1]
    print("shape.shape: {}".format(shape.shape))

    # smpl model with subj spec shape
    model = fetal_smpl_create(
        smil_data_path,
        model_type="smpl",
        gender="infant",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="npz",
        v_template=shape,
    )
    model.to(device)

    # move to torch tensor
    _to_pt_float_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    kpt_seq = _to_pt_float_tensor(kpt_seq)
    segm_vertex_seq = [_to_pt_float_tensor(v) for v in segm_vertex_seq]
    body_pose_seq = _to_pt_float_tensor(body_pose_seq)
    transl_seq = _to_pt_float_tensor(transl_seq)
    global_orient_seq = _to_pt_float_tensor(global_orient_seq)

    kpt2smpl_joint_parent_idx_map_pt = torch.tensor(
        kpt2smpl_joint_parent_idx_map, dtype=torch.long, device=device
    )

    # unpose frame by frame
    unposed_joints_list = []
    unposed_vertices_list = []
    for t in tqdm(range(kpt_seq.shape[0])):
        this_kpt = kpt_seq[t][None]
        this_segm_vertex = segm_vertex_seq[t][None]

        with torch.no_grad():
            kpt_unposed, v_unposed = unpose_keypoint_vertice(
                model,
                torch.zeros([1, 10], dtype=torch.float32, device=device),
                body_pose_seq[t : t + 1],
                transl_seq[t : t + 1],
                global_orient_seq[t : t + 1],
                this_kpt,
                this_segm_vertex,
                kpt2smpl_joint_parent_idx_map_pt,
            )
            joints = kpt_unposed.detach().cpu().numpy()
            vertices = v_unposed.detach().cpu().numpy()

        unposed_joints_list.append(joints[0])
        unposed_vertices_list.append(vertices[0])

    unposed_joints_list = np.array(unposed_joints_list)
    unposed_vertices_list = np.array(unposed_vertices_list, dtype=object)

    # save data
    save_dir = osp.join(exp_result_dir, "subj_spec", subj_name, f"{step_idx}_unposed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "segm_vertex_seq"), unposed_vertices_list)
    np.save(osp.join(save_dir, "keypoint_seq"), unposed_joints_list)
