import argparse
import os

# import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize, StandardScaler

import io_
from _base import _pick_half  # _pick_half_subs
from default_cfg import get_cfg_defaults
from pydale.estimator import CoDeLR


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Multi-source domain adaptation")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--gpus", default=None, help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    dataset = cfg.DATASET.DATASET
    atlas = cfg.DATASET.ATLAS
    sessions = cfg.DATASET.SESSIONS
    data_dir = cfg.DATASET.ROOT

    random_state = cfg.SOLVER.SEED
    connection_type = cfg.DATASET.CONNECTION
    lambda_ = cfg.SOLVER.LAMBDA_
    l2_param = cfg.SOLVER.L2PARAM
    num_repeat = cfg.DATASET.NUM_REPEAT
    run_ = cfg.DATASET.RUN

    mix_gender = cfg.DATASET.MIX_GEND
    if mix_gender:
        lambda_ = 0.0
        out_folder = "lambda0_mix_gender"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    x = []
    y = []
    genders = []

    # info_file = 'HCP_%s_half_brain_gender_equal.csv' % atlas
    info_file = '%s_%s_half_brain.csv' % (dataset, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')

    gender = info['gender'].values

    for session in sessions:
        data = io_.load_half_brain(data_dir, atlas, session, run_, connection_type, dataset=dataset)
        x.append(np.concatenate((data["Left"], data['Right'])))
        y.append(np.concatenate((np.zeros(data["Left"].shape[0]), (np.ones(data["Right"].shape[0])))))
        genders.append(np.concatenate((gender, gender)))

    x = np.concatenate(x)
    y = np.concatenate(y)
    genders = np.concatenate(genders)

    model_files = os.listdir(out_dir)
    for model_file in model_files:
        model_path = os.path.join(out_dir, model_file)
        model = torch.load(model_path)
        y_pred = model.predict(x)

        male_idx = np.where(genders == 0)
        female_idx = np.where(genders == 1)

        acc_m = accuracy_score(y[male_idx], y_pred[male_idx])
        acc_f = accuracy_score(y[female_idx], y_pred[female_idx])


if __name__ == '__main__':
    main()
