import copy
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
from pydale.estimator import GSLR

BASE_RESULT_DICT = {'pred_loss': [], 'hsic_loss': [], 'lambda': [], 'split': [], 'fold': [], 'train_gender': [],
                    'time_used': []}


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
        out_folder = "lambda0_mix_gender"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = dict()
    # groups = dict()

    info_file = '%s_%s_half_brain.csv' % (cfg.DATASET.DATASET, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')
    group_label = info['gender'].values

    if dataset == "hcp":
        sessions = ['REST1', 'REST2']
        for session in sessions:
            data[session] = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)
            # groups[session] = {'gender': group_label}

        if test_size == 0:
            results = run_no_sub_hold_hcp(data, group_label, lambda_, l2_param, mix_group, out_dir, num_repeat, random_state)

    elif dataset == "gsp":
        print()
    else:
        raise ValueError("Invalid dataset %s" % dataset)

    out_filename = 'results_%s_L%s_test_size_%s_%s' % (dataset, int(lambda_), str(int(test_size * 10)), run_, random_state)


def train_modal(lambda_, l2_param, x_train_fold, fit_kws, out_dir, model_filename):
    model_path = os.path.join(out_dir, "%s.pt" % model_filename)

    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = GSLR(lambda_=lambda_, C=l2_param, max_iter=5000)
        model.fit(x_train_fold, **fit_kws)
        torch.save(model, model_path)

    return model


def save_loop_results(model, res_dict, xy_test, train_group, train_session, lambda_, i_split, i_fold):
    for acc_key in xy_test:
        test_x, test_y = xy_test[acc_key]
        y_pred_ = model.predict(test_x)
        acc_ = accuracy_score(test_y, y_pred_)
        res_dict[acc_key].append(acc_)

    res_dict['pred_loss'].append(model.losses['pred'][-1])
    res_dict['code_loss'].append(model.losses['code'][-1])

    # res['n_iter'].append(n_iter)
    # n_iter += 1
    res_dict['train_gender'].append(train_group)
    res_dict['train_session'].append(train_session)
    res_dict['split'].append(i_split)
    res_dict['fold'].append(i_fold)

    res_dict['lambda'].append(lambda_)
    # res['test_size'].append(test_size)
    res_dict['time_used'].append(model.losses['time'][-1])

    return res_dict


def run_no_sub_hold_hcp(data, groups, lambda_, l2_param, mix_group, out_dir, num_repeat, random_state):
    # main loop
    res = {**{"acc_ic_is": [], "acc_ic_os": [], "acc_oc_is": [], "acc_oc_os": [], 'train_session': []},
           **copy.deepcopy(BASE_RESULT_DICT)}

    # for test_size in test_sizes:
    #     spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    for train_session, test_session in [('REST1', 'REST2'), ('REST2', 'REST1')]:
        groups_train_session = np.copy(groups[train_session]['gender'].reshape((-1, 1)))
        groups_test_session = np.copy(groups[test_session]['gender'].reshape((-1, 1)))

        for i_split in range(num_repeat):
            x_all = dict()
            y_all = dict()
            for session in [train_session, test_session]:
                x, y, x1, y1 = _pick_half(data[session], random_state=random_state * (i_split + 1))
                # x, y = _pick_half_subs(data[run_])
                y = label_binarize(y, classes=[-1, 1]).reshape(-1)
                y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

                x_all[session] = [x, x1]
                y_all[session] = [y, y1]

            for i_fold in [0, 1]:
                x_train_fold = x_all[train_session][i_fold]
                y_train_fold = y_all[train_session][i_fold]

                # scaler = StandardScaler()
                # scaler.fit(x_train_fold)
                for train_group in [0, 1]:
                    tgt_is_idx = np.where(groups_train_session == train_group)[0]
                    nt_is_idx = np.where(groups_train_session == 1 - train_group)[0]

                    tgt_os_idx = np.where(groups_test_session == train_group)[0]
                    nt_os_idx = np.where(groups_test_session == 1 - train_group)[0]

                    if mix_group:
                        if train_group == 1:
                            continue
                        train_group = "mix"

                    xy_test = {"acc_ic_is": [x_all[train_session][1 - i_fold][tgt_is_idx],
                                             y_all[train_session][1 - i_fold][tgt_is_idx]],
                               "acc_ic_os": [np.concatenate([x_all[test_session][i][tgt_os_idx] for i in range(2)]),
                                             np.concatenate([y_all[test_session][i][tgt_os_idx] for i in range(2)])],
                               "acc_oc_is": [x_all[train_session][1 - i_fold][nt_is_idx],
                                             y_all[train_session][1 - i_fold][nt_is_idx]],
                               "acc_oc_os": [np.concatenate([x_all[test_session][i][nt_os_idx] for i in range(2)]),
                                             np.concatenate([y_all[test_session][i][nt_os_idx] for i in range(2)])]}

                    fit_kws = {"y": y_train_fold[tgt_is_idx], "groups": groups_train_session, "target_idx": tgt_is_idx}
                    model_filename = "lambda%s_%s_%s_%s_gender_%s_%s" % (int(lambda_), train_session, i_split, i_fold,
                                                                         train_group, random_state)
                    if mix_group:
                        model_filename = model_filename + "_mix_gender"
                        fit_kws = {"y": y_train_fold, "groups": groups_train_session, "target_idx": None}

                    model = train_modal(lambda_, l2_param, x_train_fold, fit_kws, out_dir, model_filename)
                    res = save_loop_results(model, res, xy_test, train_group, train_session, lambda_, i_split, i_fold)

    return res


def run_sub_hold_hcp(data, groups, lambda_, l2_param, mix_group, out_dir, num_repeat, test_size, random_state):
    res = {**{"acc_ic_is": [], "acc_ic_os": [], "acc_oc_is": [], "acc_oc_os": [], "acc_tgt_test_sub": [],
              "acc_nt_test_sub": [], 'train_session': []},
           **copy.deepcopy(BASE_RESULT_DICT)}
    # for test_size in test_sizes:
    #     spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    for train_session, test_session in [('REST1', 'REST2'), ('REST2', 'REST1')]:
        # groups_train_session = np.copy(groups[train_session]['gender'].reshape((-1, 1)))
        # groups_test_session = np.copy(groups[test_session]['gender'].reshape((-1, 1)))
        x_all = dict()
        y_all = dict()
        sss = StratifiedShuffleSplit(n_splits=num_repeat, test_size=test_size, random_state=random_state)
        for session in [train_session, test_session]:
            x, y, x1, y1 = _pick_half(data[session], random_state=random_state)
            y = label_binarize(y, classes=[-1, 1]).reshape(-1)
            y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

        x_all[session] = [x, x1]
        y_all[session] = [y, y1]

        for i_split, (train_sub, test_sub) in enumerate(sss.split(x, groups)):
            for train_fold in [0, 1]:
                x_train_fold = x_all[train_session][train_fold]
                y_train_fold = y_all[train_session][train_fold]
                x_test_fold = x_all[1 - train_fold]
                y_test_fold = y_all[1 - train_fold]

                for target_group in [0, 1]:
                    train_sub_tgt_idx = np.where(groups[train_sub] == target_group)[0]
                    train_sub_nt_idx = np.where(groups[train_sub] == 1 - target_group)[0]
                    test_sub_tgt_idx = np.where(groups[test_sub] == target_group)[0]
                    test_sub_nt_idx = np.where(groups[test_sub] == 1 - target_group)[0]

                    if mix_group:
                        if train_group == 1:
                            continue
                        train_group = "mix"

                    x_train_fold_test_tgt = x_test_fold[train_sub][train_sub_tgt_idx]
                    y_train_fold_test_tgt = y_test_fold[train_sub][train_sub_tgt_idx]
                    x_train_fold_test_nt = x_test_fold[train_sub][train_sub_nt_idx]
                    y_train_fold_test_nt = y_test_fold[train_sub][train_sub_nt_idx]

                    x_train = x_train_fold[train_sub]

                    xy_test = {"acc_ic_is": [x_all[train_session][train_sub][train_sub_tgt_idx],
                                             y_all[train_session][train_sub][train_sub_tgt_idx]],
                               "acc_ic_os": [np.concatenate([x_all[test_session][i][train_sub_tgt_idx]
                                                             for i in range(2)]),
                                             np.concatenate([y_all[test_session][i][train_sub_tgt_idx]
                                                             for i in range(2)])],
                               "acc_oc_is": [x_all[train_session][train_sub][train_sub_nt_idx],
                                             y_all[train_session][train_sub][train_sub_nt_idx]],
                               "acc_oc_os": [np.concatenate([x_all[test_session][i][train_sub_nt_idx] for i in range(2)]),
                                             np.concatenate([y_all[test_session][i][nt_os_idx] for i in range(2)])],
                               "acc_tgt_test_sub": [np.concatenate((x_train_fold[test_sub][test_sub_tgt_idx],
                                                                    x_test_fold[test_sub][test_sub_tgt_idx])),
                                                    np.concatenate((y_train_fold[test_sub][test_sub_tgt_idx],
                                                                    y_test_fold[test_sub][test_sub_tgt_idx]))],
                               "acc_nt_test_sub": [np.concatenate((x_train_fold[test_sub][test_sub_nt_idx],
                                                                   x_test_fold[test_sub][test_sub_nt_idx])),
                                                   np.concatenate((y_train_fold[test_sub][test_sub_nt_idx],
                                                                   y_test_fold[test_sub][test_sub_nt_idx]))]}
                    model_filename = model_filename + "_test_sub_0%s" % str(int(test_size * 10))
                    fit_kws = {"y": y_train_fold[train_sub][train_sub_tgt_idx], "groups": groups[train_sub],
                               "target_idx": train_sub_tgt_idx}




    # out_filename = 'results_%s_L%s_test_size_%s_%s' % (dataset, int(lambda_), str(int(test_size * 10)), run_, random_state)
    #
    # out_filename = 'results_%s_lambda%s_test_sub_0%s_%s_%s' % (dataset, int(lambda_), str(int(test_size * 10)), run_,
    #                                                            random_state)
