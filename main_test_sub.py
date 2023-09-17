import argparse
import os

# import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize, StandardScaler

import io_
from _base import _pick_half  # _pick_half_subs
from default_cfg import get_cfg_defaults
from pydale.estimator import GSLR


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(
        description="Hold-out part of subjects for testing"
    )
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

    atlas = cfg.DATASET.ATLAS
    data_dir = cfg.DATASET.ROOT
    dataset = cfg.DATASET.DATASET

    random_state = cfg.SOLVER.SEED
    connection_type = cfg.DATASET.CONNECTION
    lambda_ = cfg.SOLVER.LAMBDA_
    l2_param = cfg.SOLVER.L2PARAM
    num_repeat = cfg.DATASET.NUM_REPEAT
    run_ = cfg.DATASET.RUN
    test_size = cfg.DATASET.TEST_SIZE
    mix_group = cfg.DATASET.MIX_GROUP
    if mix_group:
        lambda_ = 0.0
        out_folder = "lambda0_mix_gender"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # test_sizes = [0.1, 0.2, 0.3, 0.4]

    info_file = "%s_%s_half_brain.csv" % (cfg.DATASET.DATASET, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col="ID")

    data = io_.load_half_brain(
        data_dir,
        atlas,
        session=None,
        run=run_,
        connection_type=connection_type,
        dataset=dataset,
    )
    groups = info["gender"].values

    # main loop
    res = {
        "acc_ic": [],
        "acc_oc": [],
        "pred_loss": [],
        "hsic_loss": [],
        "lambda": [],
        "split": [],
        "fold": [],
        "train_gender": [],
        "time_used": [],
        "test_size": [],
    }
    if 0 < test_size < 1:
        res["acc_tgt_test_sub"] = []
        res["acc_nt_test_sub"] = []

    x, y, x1, y1 = _pick_half(data, random_state=random_state)
    y = label_binarize(y, classes=[-1, 1]).reshape(-1)
    y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

    x_all = [x, x1]
    y_all = [y, y1]

    sss = StratifiedShuffleSplit(
        n_splits=num_repeat, test_size=test_size, random_state=random_state
    )
    for i_split, (train_sub, test_sub) in enumerate(sss.split(x, groups)):
        for train_fold in [0, 1]:
            x_train_fold = x_all[train_fold]
            y_train_fold = y_all[train_fold]
            x_test_fold = x_all[1 - train_fold]
            y_test_fold = y_all[1 - train_fold]

            for target_group in [0, 1]:
                train_sub_tgt_idx = np.where(groups[train_sub] == target_group)[0]
                train_sub_nt_idx = np.where(groups[train_sub] == 1 - target_group)[0]
                test_sub_tgt_idx = np.where(groups[test_sub] == target_group)[0]
                test_sub_nt_idx = np.where(groups[test_sub] == 1 - target_group)[0]

                if mix_group:
                    target_group = "mix"
                    if target_group == 1:
                        continue

                x_train_fold_test_tgt = x_test_fold[train_sub][train_sub_tgt_idx]
                y_train_fold_test_tgt = y_test_fold[train_sub][train_sub_tgt_idx]
                x_train_fold_test_nt = x_test_fold[train_sub][train_sub_nt_idx]
                y_train_fold_test_nt = y_test_fold[train_sub][train_sub_nt_idx]

                model = GSLR(lambda_=lambda_, C=l2_param, max_iter=5000)
                fit_kws = {
                    "y": y_train_fold[train_sub_tgt_idx],
                    "groups": groups,
                    "target_idx": train_sub_tgt_idx,
                }
                model_filename = "%s_lambda%s_%s_%s_gender_%s_%s" % (
                    dataset,
                    int(lambda_),
                    i_split,
                    train_fold,
                    target_group,
                    random_state,
                )
                if 0 < test_size < 1:
                    x_train = x_train_fold[train_sub]
                    xy_test = {
                        "acc_ic": [x_train_fold_test_tgt, y_train_fold_test_tgt],
                        "acc_oc": [x_train_fold_test_nt, y_train_fold_test_nt],
                        "acc_tgt_test_sub": [
                            np.concatenate(
                                (
                                    x_train_fold[test_sub][test_sub_tgt_idx],
                                    x_test_fold[test_sub][test_sub_tgt_idx],
                                )
                            ),
                            np.concatenate(
                                (
                                    y_train_fold[test_sub][test_sub_tgt_idx],
                                    y_test_fold[test_sub][test_sub_tgt_idx],
                                )
                            ),
                        ],
                        "acc_nt_test_sub": [
                            np.concatenate(
                                (
                                    x_train_fold[test_sub][test_sub_nt_idx],
                                    x_test_fold[test_sub][test_sub_nt_idx],
                                )
                            ),
                            np.concatenate(
                                (
                                    y_train_fold[test_sub][test_sub_nt_idx],
                                    y_test_fold[test_sub][test_sub_nt_idx],
                                )
                            ),
                        ],
                    }
                    model_filename = model_filename + "_test_sub_0%s" % str(
                        int(test_size * 10)
                    )
                    fit_kws = {
                        "y": y_train_fold[train_sub][train_sub_tgt_idx],
                        "groups": groups[train_sub],
                        "target_idx": train_sub_tgt_idx,
                    }
                else:
                    x_train = x_train_fold
                    xy_test = {
                        "acc_ic": [x_train_fold_test_tgt, y_train_fold_test_tgt],
                        "acc_oc": [x_train_fold_test_nt, y_train_fold_test_nt],
                    }

                if mix_group:
                    model_filename = model_filename + "_mix_gender"
                    if 0 < test_size < 1:
                        fit_kws = {
                            "y": y_train_fold[train_sub],
                            "group": groups[train_sub],
                            "target_idx": None,
                        }
                    else:
                        fit_kws = {
                            "y": y_train_fold,
                            "group": groups,
                            "target_idx": None,
                        }

                model_path = os.path.join(out_dir, "%s.pt" % model_filename)

                if os.path.exists(model_path):
                    model = torch.load(model_path)
                else:
                    model.fit(x_train, **fit_kws)
                    torch.save(model, model_path)

                for acc_key in xy_test:
                    test_x, test_y = xy_test[acc_key]
                    y_pred_ = model.predict(test_x)
                    acc_ = accuracy_score(test_y, y_pred_)
                    res[acc_key].append(acc_)

                res["pred_loss"].append(model.losses["pred"][-1])
                res["hsic_loss"].append(model.losses["hsic"][-1])

                res["train_gender"].append(target_group)
                res["split"].append(i_split)
                res["fold"].append(train_fold)

                res["lambda"].append(lambda_)
                res["test_size"].append(test_size)
                res["time_used"].append(model.losses["time"][-1])

    res_df = pd.DataFrame.from_dict(res)

    out_filename = "results_%s_lambda%s_test_sub_0%s_%s_%s" % (
        dataset,
        int(lambda_),
        str(int(test_size * 10)),
        run_,
        random_state,
    )
    if mix_group:
        out_filename = out_filename + "_mix_gender"
    out_file = os.path.join(out_dir, "%s.csv" % out_filename)
    res_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
