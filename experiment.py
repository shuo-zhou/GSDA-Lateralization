import copy
import os

# import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score  # , roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import io_
from _base import _pick_half  # _pick_half_subs
from pydale.estimator import GSLR

BASE_RESULT_DICT = {
    "pred_loss": [],
    "hsic_loss": [],
    "lambda": [],
    "split": [],
    "fold": [],
    "train_gender": [],
    "time_used": [],
}


def run_experiment(cfg):
    atlas = cfg.DATASET.ATLAS
    connection_type = cfg.DATASET.CONNECTION
    data_dir = cfg.DATASET.ROOT
    dataset = cfg.DATASET.DATASET
    lambda_ = cfg.SOLVER.LAMBDA_
    run_ = cfg.DATASET.RUN
    random_state = cfg.SOLVER.SEED
    test_size = cfg.DATASET.TEST_SIZE

    l2_param = cfg.SOLVER.L2PARAM
    num_repeat = cfg.DATASET.NUM_REPEAT
    mix_group = cfg.DATASET.MIX_GROUP

    if mix_group:
        lambda_ = 0.0
        out_folder = "lambda0_group_mix"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    info_file = "%s_%s_half_brain.csv" % (cfg.DATASET.DATASET, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col="ID")
    group_label = info["gender"].values

    kwargs = {
        "groups": group_label,
        "lambda_": lambda_,
        "l2_param": l2_param,
        "mix_group": mix_group,
        "out_dir": out_dir,
        "num_repeat": num_repeat,
        "test_size": test_size,
        "random_state": random_state,
    }

    if dataset == "HCP":
        data = dict()
        sessions = ["REST1", "REST2"]
        for session in sessions:
            data[session] = io_.load_half_brain(
                data_dir, atlas, session, run_, connection_type
            )
        kwargs = {**{"data": data}, **kwargs}
        if test_size == 0:
            results = run_no_sub_hold_hcp(**kwargs)
        elif 0 < test_size < 1:
            results = run_sub_hold_hcp(**kwargs)
        else:
            raise ValueError("Invalid test_size %s" % test_size)

    elif dataset == "gsp":
        data = io_.load_half_brain(
            data_dir,
            atlas,
            session=None,
            run=run_,
            connection_type=connection_type,
            dataset=dataset,
        )
        kwargs = {**{"data": data}, **kwargs}
        if 0 < test_size < 1:
            results = run_sub_hold_gsp(**kwargs)
        elif test_size == 0:
            results = run_sub_hold_gsp(**kwargs)
        else:
            raise ValueError("Invalid test_size %s" % test_size)
    else:
        raise ValueError("Invalid dataset %s" % dataset)

    res_df = pd.DataFrame.from_dict(results)
    out_filename = "results_%s_L%s_test_size0%s_%s_%s" % (
        dataset,
        int(lambda_),
        str(int(test_size * 10)),
        run_,
        random_state,
    )

    if mix_group:
        out_filename = out_filename + "_group_mix"
    out_file = os.path.join(out_dir, "%s.csv" % out_filename)

    return res_df, out_file


def train_modal(lambda_, l2_param, x_train, fit_kws, out_dir, model_filename):
    model_path = os.path.join(out_dir, "%s.pt" % model_filename)

    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = GSLR(lambda_=lambda_, C=l2_param, max_iter=5000)
        model.fit(x_train, **fit_kws)
        torch.save(model, model_path)

    return model


def save_loop_results(
    model,
    res_dict,
    xy_test,
    train_group,
    train_session=None,
    lambda_=0.0,
    i_split=0,
    train_fold=0,
):
    for acc_key in xy_test:
        test_x, test_y = xy_test[acc_key]
        y_pred_ = model.predict(test_x)
        acc_ = accuracy_score(test_y, y_pred_)
        res_dict[acc_key].append(acc_)

    res_dict["pred_loss"].append(model.losses["pred"][-1])
    if "hsic" in model.losses:
        res_dict["hsic_loss"].append(model.losses["hsic"][-1])
    else:
        res_dict["hsic_loss"].append(model.losses["code"][-1])

    # res['n_iter'].append(n_iter)
    # n_iter += 1
    res_dict["train_gender"].append(train_group)
    if train_session is not None:
        res_dict["train_session"].append(train_session)
    else:
        res_dict["train_session"].append(0)
    res_dict["split"].append(i_split)
    res_dict["fold"].append(train_fold)

    res_dict["lambda"].append(lambda_)
    # res['test_size'].append(test_size)
    res_dict["time_used"].append(model.losses["time"][-1])

    return res_dict


def run_no_sub_hold_hcp(
    data,
    groups,
    lambda_,
    l2_param,
    mix_group,
    out_dir,
    num_repeat,
    random_state,
    **kwargs
):
    # main loop
    res = {
        **{
            "acc_ic_is": [],
            "acc_ic_os": [],
            "acc_oc_is": [],
            "acc_oc_os": [],
            "train_session": [],
        },
        **copy.deepcopy(BASE_RESULT_DICT),
    }

    for train_session, test_session in [("REST1", "REST2"), ("REST2", "REST1")]:

        for i_split in range(num_repeat):
            x_all = dict()
            y_all = dict()
            for session in [train_session, test_session]:
                x, y, x1, y1 = _pick_half(
                    data[session], random_state=random_state * (i_split + 1)
                )
                # x, y = _pick_half_subs(data[run_])

                x_all[session] = [x, x1]
                y_all[session] = [y, y1]

            for i_fold in [0, 1]:
                x_train_fold = x_all[train_session][i_fold]
                y_train_fold = y_all[train_session][i_fold]

                # scaler = StandardScaler()
                # scaler.fit(x_train_fold)
                for train_group in [0, 1]:
                    tgt_is_idx = np.where(groups == train_group)[0]
                    nt_is_idx = np.where(groups == 1 - train_group)[0]

                    tgt_os_idx = np.where(groups == train_group)[0]
                    nt_os_idx = np.where(groups == 1 - train_group)[0]

                    if mix_group:
                        if train_group == 1:
                            continue
                        train_group = "mix"

                    xy_test = {
                        "acc_ic_is": [
                            x_all[train_session][1 - i_fold][tgt_is_idx],
                            y_all[train_session][1 - i_fold][tgt_is_idx],
                        ],
                        "acc_ic_os": [
                            np.concatenate(
                                [x_all[test_session][i][tgt_os_idx] for i in range(2)]
                            ),
                            np.concatenate(
                                [y_all[test_session][i][tgt_os_idx] for i in range(2)]
                            ),
                        ],
                        "acc_oc_is": [
                            x_all[train_session][1 - i_fold][nt_is_idx],
                            y_all[train_session][1 - i_fold][nt_is_idx],
                        ],
                        "acc_oc_os": [
                            np.concatenate(
                                [x_all[test_session][i][nt_os_idx] for i in range(2)]
                            ),
                            np.concatenate(
                                [y_all[test_session][i][nt_os_idx] for i in range(2)]
                            ),
                        ],
                    }

                    fit_kws = {
                        "y": y_train_fold[tgt_is_idx],
                        "groups": groups,
                        "target_idx": tgt_is_idx,
                    }
                    model_filename = "HCP_L%s_test_size00_%s_%s_%s_group_%s_%s" % (
                        int(lambda_),
                        train_session,
                        i_split,
                        i_fold,
                        train_group,
                        random_state,
                    )
                    if mix_group:
                        model_filename = model_filename + "_group_mix"
                        fit_kws = {
                            "y": y_train_fold,
                            "groups": groups,
                            "target_idx": None,
                        }

                    model = train_modal(
                        lambda_,
                        l2_param,
                        x_train_fold,
                        fit_kws,
                        out_dir,
                        model_filename,
                    )
                    res = save_loop_results(
                        model,
                        res,
                        xy_test,
                        train_group,
                        train_session,
                        lambda_,
                        i_split,
                        i_fold,
                    )

    return res


def run_sub_hold_hcp(
    data,
    groups,
    lambda_,
    l2_param,
    mix_group,
    out_dir,
    num_repeat,
    test_size,
    random_state,
):
    res = {
        **{
            "acc_ic_is": [],
            "acc_ic_os": [],
            "acc_oc_is": [],
            "acc_oc_os": [],
            "acc_tgt_test_sub": [],
            "acc_nt_test_sub": [],
            "train_session": [],
        },
        **copy.deepcopy(BASE_RESULT_DICT),
    }
    x_all = dict()
    y_all = dict()

    for session in ["REST1", "REST2"]:
        x, y, x1, y1 = _pick_half(data[session], random_state=random_state)

        x_all[session] = [x, x1]
        y_all[session] = [y, y1]

    for train_session, test_session in [("REST1", "REST2"), ("REST2", "REST1")]:
        sss = StratifiedShuffleSplit(
            n_splits=num_repeat, test_size=test_size, random_state=random_state
        )

        for i_split, (train_sub, test_sub) in enumerate(sss.split(x, groups)):
            for train_fold in [0, 1]:
                x_train_fold = x_all[train_session][train_fold]
                y_train_fold = y_all[train_session][train_fold]

                for target_group in [0, 1]:
                    train_sub_tgt_idx = np.where(groups[train_sub] == target_group)[0]
                    train_sub_nt_idx = np.where(groups[train_sub] == 1 - target_group)[
                        0
                    ]
                    test_sub_tgt_idx = np.where(groups[test_sub] == target_group)[0]
                    test_sub_nt_idx = np.where(groups[test_sub] == 1 - target_group)[0]

                    if mix_group:
                        if target_group == 1:
                            continue
                        target_group = "mix"

                    xy_test = {
                        "acc_ic_is": [
                            x_all[train_session][1 - train_fold][train_sub][
                                train_sub_tgt_idx
                            ],
                            y_all[train_session][1 - train_fold][train_sub][
                                train_sub_tgt_idx
                            ],
                        ],
                        "acc_ic_os": [
                            np.concatenate(
                                [
                                    x_all[test_session][i][train_sub][train_sub_tgt_idx]
                                    for i in range(2)
                                ]
                            ),
                            np.concatenate(
                                [
                                    y_all[test_session][i][train_sub][train_sub_tgt_idx]
                                    for i in range(2)
                                ]
                            ),
                        ],
                        "acc_oc_is": [
                            x_all[train_session][1 - train_fold][train_sub][
                                train_sub_nt_idx
                            ],
                            y_all[train_session][1 - train_fold][train_sub][
                                train_sub_nt_idx
                            ],
                        ],
                        "acc_oc_os": [
                            np.concatenate(
                                [
                                    x_all[test_session][i][train_sub][train_sub_nt_idx]
                                    for i in range(2)
                                ]
                            ),
                            np.concatenate(
                                [
                                    y_all[test_session][i][train_sub][train_sub_nt_idx]
                                    for i in range(2)
                                ]
                            ),
                        ],
                        "acc_tgt_test_sub": [[], []],
                        "acc_nt_test_sub": [[], []],
                    }

                    for _s in [train_session, test_session]:
                        for _fold in (0, 1):
                            xy_test["acc_tgt_test_sub"][0].append(
                                x_all[_s][_fold][test_sub][test_sub_tgt_idx]
                            )
                            xy_test["acc_tgt_test_sub"][1].append(
                                y_all[_s][_fold][test_sub][test_sub_tgt_idx]
                            )

                            xy_test["acc_nt_test_sub"][0].append(
                                x_all[_s][_fold][test_sub][test_sub_nt_idx]
                            )
                            xy_test["acc_nt_test_sub"][1].append(
                                y_all[_s][_fold][test_sub][test_sub_nt_idx]
                            )

                    for _key in ["acc_tgt_test_sub", "acc_nt_test_sub"]:
                        for _idx in range(2):
                            xy_test[_key][_idx] = np.concatenate(xy_test[_key][_idx])

                    fit_kws = {
                        "y": y_train_fold[train_sub][train_sub_tgt_idx],
                        "groups": groups[train_sub],
                        "target_idx": train_sub_tgt_idx,
                    }
                    model_filename = "HCP_L%s_test_size0%s_%s_%s_group_%s_%s" % (
                        int(lambda_),
                        str(int(test_size * 10)),
                        i_split,
                        train_fold,
                        target_group,
                        random_state,
                    )
                    x_train = x_train_fold[train_sub]

                    if mix_group:
                        model_filename = model_filename + "_group_mix"
                        fit_kws = {
                            "y": y_train_fold[train_sub],
                            "groups": groups[train_sub],
                            "target_idx": None,
                        }

                    model = train_modal(
                        lambda_, l2_param, x_train, fit_kws, out_dir, model_filename
                    )
                    res = save_loop_results(
                        model,
                        res,
                        xy_test,
                        target_group,
                        train_session,
                        lambda_,
                        i_split,
                        train_fold,
                    )

    return res


def run_sub_hold_gsp(
    data,
    groups,
    lambda_,
    l2_param,
    mix_group,
    out_dir,
    num_repeat,
    test_size,
    random_state,
):
    res = {
        **{"acc_ic": [], "acc_oc": [], "acc_tgt_test_sub": [], "acc_nt_test_sub": []},
        **copy.deepcopy(BASE_RESULT_DICT),
    }

    x, y, x1, y1 = _pick_half(data, random_state=random_state)

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

                fit_kws = {
                    "y": y_train_fold[train_sub_tgt_idx],
                    "groups": groups,
                    "target_idx": train_sub_tgt_idx,
                }
                model_filename = "gsp_L%s_test_size0%s_%s_%s_group_%s_%s" % (
                    int(lambda_),
                    str(int(test_size * 10)),
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
                    model_filename = model_filename + "_group_mix"
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

                model = train_modal(
                    lambda_, l2_param, x_train, fit_kws, out_dir, model_filename
                )
                res = save_loop_results(
                    model,
                    res,
                    xy_test,
                    target_group,
                    lambda_=lambda_,
                    i_split=i_split,
                    train_fold=train_fold,
                )

    return res


def run_no_sub_hold_gsp(
    data,
    groups,
    lambda_,
    l2_param,
    mix_group,
    out_dir,
    num_repeat,
    random_state,
    **kwargs
):
    res = {
        **{"acc_ic": [], "acc_oc": []},
        **copy.deepcopy(BASE_RESULT_DICT),
    }
    for i_split in range(num_repeat):
        x, y, x1, y1 = _pick_half(data, random_state=random_state * (i_split + 1))
        x_all = [x, x1]
        y_all = [y, y1]

        for i_fold in [0, 1]:
            x_train = x_all[i_fold]
            y_train = y_all[i_fold]

            # scaler = StandardScaler()
            # scaler.fit(x_train)
            for train_group in [0, 1]:
                tgt_idx = np.where(groups == train_group)[0]
                nt_idx = np.where(groups == 1 - train_group)[0]

                if mix_group:
                    train_group = "mix"
                    if train_group == 1:
                        continue

                xy_test = {
                    "acc_ic": [x_all[1 - i_fold][tgt_idx], x_all[1 - i_fold][tgt_idx]],
                    "acc_oc": [x_all[1 - i_fold][nt_idx], x_all[1 - i_fold][nt_idx]],
                }

                fit_kws = {
                    "y": y_train[tgt_idx],
                    "groups": groups,
                    "target_idx": tgt_idx,
                }
                model_filename = "gsp_L%s_test_size00_%s_%s_group_%s_%s" % (
                    int(lambda_),
                    i_split,
                    i_fold,
                    train_group,
                    random_state,
                )

                if mix_group:
                    model_filename = model_filename + "_group_mix"
                    fit_kws = {
                        "y": y_train,
                        "groups": groups,
                        "target_idx": None,
                    }

                model = train_modal(
                    lambda_, l2_param, x_train, fit_kws, out_dir, model_filename
                )
                res = save_loop_results(
                    model,
                    res,
                    xy_test,
                    train_group,
                    lambda_=lambda_,
                    i_split=i_split,
                    train_fold=i_fold,
                )

    return res