#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/22/2024
#
# Distributed under terms of the MIT license.

"""Evaluation step for fetal smil model

apply the same process (0-2, 0-3, 1-1) as training. but use our model.

1. init align smpl to posed and unpose (subj specific)
2. init smpl beta from unposed (subj specific)
3. align subj spec shape to posed (subj specific)

The folder structure:

/exp_{{name_of_exp}}
        /evaluation
                /{{step_idx_of_model}}/{{num_beta}}/{{folder_name}}
                    /subj_spec
                        /{{subj_name}}
                            /init_posed
                                /transl_seq_his (0-2)
                                /global_orient_seq_his (0-2)
                                /body_pose_seq_his (0-2)
                                /beta_his (0-2)
                            /init_unposed
                                /segm_vertex_seq (0-2)
                                /keypoint_seq (0-2)
                                /beta_his (0-3)
                                /shape_his (0-3)
                            /1_posed
                                /transl_seq_his (1-1)
                                /global_orient_seq_his (1-1)
                                /body_pose_seq_his (1-1)v
"""

import argparse
import json
import os
import os.path as osp

from fetal_smpl.alignment import (
    main_align_subj_spec_shape_to_posed,
    main_init_align_smpl_to_posed_and_unpose,
)
from fetal_smpl.learn_shape import main_init_smpl_beta_from_unposed


def _read_data_split(
    data_split, data_split_cfg_dir
) -> dict[str, dict[str, tuple[int, int]]]:
    """Read data split config file.
    Return folder_name -> subj_name -> (start_frame, end_frame)
    """

    cfg_dir = osp.join(data_split_cfg_dir, f"{data_split}")
    assert osp.exists(cfg_dir), f"{cfg_dir} does not exist"

    filename_list = sorted(os.listdir(cfg_dir))

    data_split_cfg = {}
    for filename in filename_list:
        folder_name = filename.replace(".json", "")
        with open(osp.join(cfg_dir, filename), "r") as f:
            subj_name2frame = json.load(f)
        data_split_cfg[folder_name] = subj_name2frame

    return data_split_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--step-idx",
        type=str,
        help="The step idx of optimization process. "
        "If , separated, all steps are repeated for "
        "all step-name excpet the data preparation and init steps. "
        "When 0, use the original smil model.",
    )
    parser.add_argument(
        "--num-beta",
        type=str,
        default="10",
        help="The number of betas to use. If , separated, "
        "all steps are repeated for all num-betas",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./results/smil_fetal_exp_test",
        help="The name of the exp dir",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        help="The split of subjects to use",
    )
    parser.add_argument(
        "--data-split-cfg-dir",
        type=str,
        default="data_split",
        help="Dir containing the data split cfg",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="local",
        help="The runner to use",
    )
    parser.add_argument(
        "--folder-name",
        type=str,
        default="train",
        help="The folder name to use",
    )
    parser.add_argument(
        "--scale-body-size",
        type=float,
        default=1.0,
        help="The scale factor to resize the body size",
    )
    args = parser.parse_args()

    def _run_task(task_list, runner_type: str, submitit_log_dir=None, partition=None):
        if runner_type == "local":
            for task in task_list:
                task[0](*task[1:])
        else:
            raise ValueError(f"Unknown runner type: {runner_type}")

    # two coordinate systems
    # pixel/image coordinate
    # physical coordinate (origin aligned with pixel's)
    resolution = 0.004  # pixel to meter

    scale_body_size = args.scale_body_size
    if not 0 < scale_body_size < 2:
        raise ValueError(f"scale_body_size should be in (0, 1), got {scale_body_size}")

    smil_data_path = "./models"

    submitit_log_dir = osp.join(args.exp_dir, "submitit_logs")

    data_split_cfg = _read_data_split(args.data_split, args.data_split_cfg_dir)
    assert (
        args.folder_name in data_split_cfg
    ), f"{args.folder_name} not in {data_split_cfg}"

    exp_data_dir = osp.join(args.exp_dir, "data")
    exp_data_split_dir = osp.join(exp_data_dir, args.folder_name)
    subj_name_list = list(data_split_cfg[args.folder_name].keys())

    idx_step_of_model_list = [int(i) for i in args.step_idx.split(",")]
    num_betas_list = [int(i) for i in args.num_beta.split(",")]
    for idx_step_of_model in idx_step_of_model_list:
        if idx_step_of_model == 0:
            fetal_smpl_data_dict_path = None  # use original smil model
        else:
            fetal_smpl_data_dict_path = osp.join(
                args.exp_dir, "training", "model", f"step_{idx_step_of_model}.npy"
            )
        for num_betas in num_betas_list:
            exp_result_dir = osp.join(
                args.exp_dir,
                "evaluation",
                f"model_{idx_step_of_model}",
                f"num_betas_{num_betas:02d}",
                args.folder_name,
            )
            os.makedirs(exp_result_dir, exist_ok=True)

            # init_align_smil_to_posed_and_unpose
            task_list = []
            for subj_name in subj_name_list:
                task_list.append(
                    (
                        main_init_align_smpl_to_posed_and_unpose,
                        exp_data_split_dir,
                        exp_result_dir,
                        subj_name,
                        smil_data_path,
                        num_betas,
                        fetal_smpl_data_dict_path,
                        scale_body_size,
                    )
                )
            _s_log_dir = osp.join(
                submitit_log_dir, "init_align_smil_to_posed_and_unpose"
            )
            _run_task(task_list, args.runner, _s_log_dir)

            # init_smil_beta_from_unposed
            task_list = []
            for subj_name in subj_name_list:
                task_list.append(
                    (
                        main_init_smpl_beta_from_unposed,
                        exp_result_dir,
                        subj_name,
                        smil_data_path,
                        num_betas,
                        fetal_smpl_data_dict_path,
                        scale_body_size,
                    )
                )
            _s_log_dir = osp.join(submitit_log_dir, "init_smil_beta_from_unposed")
            _run_task(task_list, args.runner, _s_log_dir)

            # align_subj_spec_shape_to_posed
            step_idx = 1
            task_list = []
            for subj_name in subj_name_list:
                task_list.append(
                    (
                        main_align_subj_spec_shape_to_posed,
                        exp_data_split_dir,
                        exp_result_dir,
                        step_idx,
                        subj_name,
                        smil_data_path,
                        num_betas,
                        fetal_smpl_data_dict_path,
                        scale_body_size,
                    )
                )
            _s_log_dir = osp.join(
                submitit_log_dir, "align_subj_spec_shape_to_posed", f"{step_idx}"
            )
            _run_task(task_list, args.runner, _s_log_dir)
