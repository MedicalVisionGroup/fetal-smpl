#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/21/2024
#
# Distributed under terms of the MIT license.

""" """

import typing as tp

import torch

from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)
from smplx.utils import Tensor


def invert_rigid_transform_with_keypoints(
    J_transformed, rot_mats, parents, keypoints, kpt_parents=None
):
    """
    Inverts the batch_rigid_transform to recover the original joint positions.

    When unposing the keypoints, we will match the each keypoint to the closest SMPL joint
    and apply the inverse transformation of that SMPL joint to the keypoint.

    Parameters
    ----------
    J_transformed : torch.tensor BxNx3
        The transformed joint positions after applying the pose rotations.
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices used in the forward transformation.
    parents : torch.tensor N
        The kinematic tree (parent indices) of each joint.
    keypoints : torch.tensor BxN_jx3
        The posed/transformed keypoints. Each keypoints are to be matched with one
        of the joints in J_transformed. The same invert transformation will be applied
        to the keypoints.
    kpt_parents : torch.tensor N_j, optional
        The parent joint indices of each keypoint. If None, then the parent joint
        indices are inferred from the nearest joint of each keypoint. The default
        value is None.

    Returns
    -------
    joints : torch.tensor BxNx3
        The original joint positions before the transformation.
    unposed_keypoints : torch.tensor BxNx3
        The unposed keypoints.
    """
    B, N, _ = J_transformed.shape

    # Initialize tensors for joints and cumulative rotations
    joints = torch.zeros_like(J_transformed)
    R_cumulative = torch.zeros(
        B, N, 3, 3, device=J_transformed.device, dtype=J_transformed.dtype
    )

    # For the root joint (index 0)
    joints[:, 0] = J_transformed[:, 0]
    R_cumulative[:, 0] = rot_mats[:, 0]

    for i in range(1, N):
        parent = parents[i]

        # Compute the inverse of the cumulative rotation of the parent joint
        R_parent_inv = R_cumulative[:, parent].transpose(1, 2)

        # Calculate the difference in positions between the current joint and its parent
        delta_pos = J_transformed[:, i] - J_transformed[:, parent]

        # Recover the original joint position
        joints[:, i] = joints[:, parent] + torch.matmul(
            R_parent_inv, delta_pos.unsqueeze(-1)
        ).squeeze(-1)

        # Update the cumulative rotation
        R_cumulative[:, i] = torch.matmul(R_cumulative[:, parent], rot_mats[:, i])

    if kpt_parents is None:
        # nearest joint index for each keypoint
        diff = keypoints[:, :, None] - J_transformed[:, None, :]  # (B, N_j, N, 3)
        square_norm = torch.sum(diff**2, dim=-1)  # (B, N_j, N)
        idx_nn = torch.argmin(square_norm, dim=-1)  # (B, N_j)

        # assert the NN are the same for all batches
        assert torch.allclose(idx_nn[0], idx_nn, atol=1e-6, rtol=1e-6), (
            f"max abs diff: {torch.max(torch.abs(idx_nn[0] - idx_nn))}"
        )
        idx_nn = idx_nn[0]

        # parent smpl joint idx for each keypoint
        kpt_parents = parents[idx_nn]

    joints_kpt = torch.zeros_like(keypoints)
    N_j = keypoints.shape[1]
    for i in range(N_j):
        kpt = keypoints[:, i]
        parent = kpt_parents[i]

        # root joints
        if parent == -1:
            joints_kpt[:, i] = joints[:, 0]
            continue

        # Compute the inverse of the cumulative rotation of the parent joint
        R_parent_inv = R_cumulative[:, parent].transpose(1, 2)

        # For keypoints, we apply the same transformation as the joints.
        # Use joint's parent to avoid the accumulation of errors.
        delta_pos_kpt = kpt - J_transformed[:, parent]
        joints_kpt[:, i] = joints[:, parent] + torch.matmul(
            R_parent_inv, delta_pos_kpt.unsqueeze(-1)
        ).squeeze(-1)

    return joints, joints_kpt


def lbs_unpose(
    vertices: Tensor,
    keypoints: Tensor,
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
    kpt_parents: tp.Optional[Tensor] = None,
) -> tp.Tuple[Tensor, Tensor]:
    """Unpose the vertices and keypoints to the canonical space.
    Go through the same lbs process to retrieve the context data in forward process.
    Then, unpose the vertices and keypoints to the canonical space.

    When unposing the vertices, we will match the each vertex to the closest SMPL surface
    vertex and apply the inverse transformation of that SMPL vertex to the vertex.

    When unposing the keypoints, we will go through the invert lbs process of SMPL joints
    from beta and pose params (not keypoints). Apply the same inverse transformation of
    nearest joint to the keypoints.

    Parameters
    ----------
    vertices : torch.tensor B x N_v x 3
        The posed vertices of body surface. Assume global translation is subtracted
        from the vertices.
    keypoints : torch.tensor B x N_j x 3
        The posed keypoints. Assume global translation is subtracted from the keypoints.
        Assume keypoints has the same order as the SMPL joints.
    betas : torch.tensor BxNB
        The tensor of shape parameters
    pose : torch.tensor Bx(J + 1) * 3
        The pose parameters in axis-angle format
    v_template torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs : torch.tensor 1xNB
        The tensor of PCA shape displacements
    posedirs : torch.tensor Px(V * 3)
        The pose PCA coefficients
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree for the model
    lbs_weights: torch.tensor N x V x (J + 1)
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        Bx(J + 1)x9
    kpt_parents: torch.tensor N_j, optional
        The parent joint indices of each keypoint. If None, then the parent joint
        indices are inferred from the nearest joint of each keypoint. The default
        value is None.

    Returns
    -------
    keypoints_unposed : torch.tensor B x N_j x 3
        The unposed keypoints
    vertices_invert : torch.tensor B x N_v x 3
        The unposed vertices

    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(
            batch_size, -1, 3
        )

    v_posed = pose_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # Unpose the keypoints
    J_inverted, keypoints_unposed = invert_rigid_transform_with_keypoints(
        J_transformed, rot_mats, parents, keypoints, kpt_parents=kpt_parents
    )
    assert torch.allclose(J_inverted, J, atol=2e-6, rtol=1e-6), (
        f"max abs diff: {torch.max(torch.abs(J_inverted - J))}"
    )

    # 5. Do skinning:
    # W is N x V x (J + 1)
    # NOTE(YL 09/17):: given a vertices, weights are mostly zeros. Only handful are 1.
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(
        batch_size, -1, 4, 4
    )  # (B, V, 4, 4)

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )  # (B, V, 1)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # (B, V, 4)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))  # (B, V, 4, 1)

    verts = v_homo[:, :, :3, 0]  # (B, V, 3)

    # invert the SMPL verts transformation
    T_inverse = torch.inverse(T)
    verts_homo = torch.cat([verts, homogen_coord], dim=-1)  # (B, V, 4)
    verts_invert_homo = torch.matmul(T_inverse, verts_homo[..., None])  # (B, V, 4, 1)
    verts_invert = verts_invert_homo[:, :, :3, 0]

    # assert torch.allclose(
    #     verts_invert, v_posed, atol=6.5e-5, rtol=1e-6
    # ), f"max abs diff: {torch.max(torch.abs(verts_invert - v_posed))}"
    assert torch.allclose(verts_invert, v_posed, atol=0.005, rtol=1e-4), (
        f"max abs diff: {torch.max(torch.abs(verts_invert - v_posed))}"
    )

    # invert the verts transformation

    # for each vertex, match to nearest verts
    diff = vertices[:, :, None] - verts[:, None, :]  # (B, N_v, V, 3)
    square_norm = torch.sum(diff**2, dim=-1)  # (B, N_v, V)
    idx_nn = torch.argmin(square_norm, dim=-1)  # (B, N_v)
    idx_nn_expand = idx_nn[..., None, None].expand(-1, -1, 4, 4)  # (B, N_v, 4, 4)
    T_nn = torch.gather(T, dim=1, index=idx_nn_expand)  # (B, N_v, 4, 4)

    # invert by multiplying inverse transform
    T_nn_inverse = torch.inverse(T_nn)
    homogen_coord_vertices = torch.ones(
        [batch_size, vertices.shape[1], 1], dtype=dtype, device=device
    )  # (B, N_v, 1)
    vertices_homo = torch.cat([vertices, homogen_coord_vertices], dim=2)  # (B, N_v, 4)
    vertices_invert_homo = torch.matmul(
        T_nn_inverse, vertices_homo[..., None]
    )  # (B, N_v, 4, 1)
    vertices_invert = vertices_invert_homo[:, :, :3, 0]

    return keypoints_unposed, vertices_invert


def unpose_keypoint_vertice(
    model, beta, body_pose, transl, global_orient, keypoints, vertices, kpt_parents
):
    """Unpose the keypoints and vertices to the canonical space.

    Args:
    model (smplx.model): SMPL model.
    beta (torch.Tensor): Shape parameters. [B, 10]
    body_pose (torch.Tensor): Pose parameters. [B, 69]
    transl (torch.Tensor): Translation parameters. [B, 3]
    global_orient (torch.Tensor): Global orientation parameters. [B, 3]
    keypoints (torch.Tensor): Keypoints. [B, N_j, 3]
    vertices (torch.Tensor): Vertices. [B, N_v, 3]
    kpt_parents (torch.Tensor): [N_j]
        List of indices that maps keypoints to parent index in SMPL

    Returns:
    torch.Tensor: Unposed keypoints. [B, 15, 3]
    torch.Tensor: Unposed vertices. [B, 6890, 3]
    """

    # get full pose from global_orient and body_pose
    full_pose = torch.cat([global_orient, body_pose], dim=1)

    # subtract translation from vertices and keypoints
    vertices = vertices - transl[:, None, :]
    keypoints = keypoints - transl[:, None, :]

    # unpose vertices and keypoints
    keypoints_unposed, vertices_unposed = lbs_unpose(
        vertices=vertices,
        keypoints=keypoints,
        betas=beta,
        pose=full_pose,
        v_template=model.v_template,
        shapedirs=model.shapedirs,
        posedirs=model.posedirs,
        J_regressor=model.J_regressor,
        parents=model.parents,
        lbs_weights=model.lbs_weights,
        kpt_parents=kpt_parents,
    )

    return keypoints_unposed, vertices_unposed
