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

    atlas = cfg.DATASET.ATLAS
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

    sessions = ['REST1', 'REST2']  # session = 'REST1'

    test_sizes = [0.1, 0.2, 0.3, 0.4]

    data = dict()
    genders = dict()

    # info_file = 'HCP_%s_half_brain_gender_equal.csv' % atlas
    info_file = '%s_%s_half_brain.csv' % (cfg.DATASET.DATASET, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')

    gender = info['gender'].values

    for session in sessions:
        data[session] = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)
        genders[session] = {'gender': gender}

    # for seed_iter in range(50):
        # random_state = cfg.SOLVER.SEED - seed_iter

    # main loop
    res = {"acc_ic_is": [], "acc_ic_os": [], "acc_oc_is": [], "acc_oc_os": [], 'pred_loss': [], 'code_loss': [],
            'lambda': [], 'train_session': [], 'split': [], 'fold': [], 'train_gender': [], 'time_used': []}
    # for test_size in test_sizes:
    #     spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    for train_session, test_session in [('REST1', 'REST2'), ('REST2', 'REST1')]:
        genders_train = np.copy(genders[train_session]['gender'].reshape((-1, 1)))
        genders_test = np.copy(genders[test_session]['gender'].reshape((-1, 1)))

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
                x_train = x_all[train_session][i_fold]
                y_train = y_all[train_session][i_fold]
                x_test_ = x_all[train_session][1 - i_fold]
                y_test_ = y_all[train_session][1 - i_fold]
                # scaler = StandardScaler()
                # scaler.fit(x_train)
                for train_gender in [0, 1]:

                    ic_is_idx = np.where(genders_train == train_gender)[0]
                    oc_is_idx = np.where(genders_train == 1 - train_gender)[0]

                    ic_os_idx = np.where(genders_test == train_gender)[0]
                    oc_os_idx = np.where(genders_test == 1 - train_gender)[0]

                    if mix_gender:
                        train_gender = "mix"
                        if train_gender == 1:
                            continue

                    x_test_ic_is = x_test_[ic_is_idx]
                    y_test_ic_is = y_test_[ic_is_idx]
                    x_test_ic_os = np.concatenate([x_all[test_session][0][ic_os_idx],
                                                    x_all[test_session][1][ic_os_idx]])
                    y_test_ic_os = np.concatenate([y_all[test_session][0][ic_os_idx],
                                                    y_all[test_session][1][ic_os_idx]])
                    x_test_oc_is = x_test_[oc_is_idx]
                    y_test_oc_is = y_test_[oc_is_idx]
                    x_test_oc_os = np.concatenate([x_all[test_session][0][oc_os_idx],
                                                    x_all[test_session][1][oc_os_idx]])
                    y_test_oc_os = np.concatenate([y_all[test_session][0][oc_os_idx],
                                                    y_all[test_session][1][oc_os_idx]])

                    xy_test = {"acc_ic_is": [x_test_ic_is, y_test_ic_is], "acc_ic_os": [x_test_ic_os, y_test_ic_os],
                                "acc_oc_is": [x_test_oc_is, y_test_oc_is], "acc_oc_os": [x_test_oc_os, y_test_oc_os]}

                    model = CoDeLR(lambda_=lambda_, C=l2_param, max_iter=5000)
                    fit_kws = {"y": y_train[ic_is_idx], "covariates": genders_train, "target_idx": ic_is_idx}
                    model_filename = "lambda%s_%s_%s_%s_gender_%s_%s" % (int(lambda_), train_session, i_split, i_fold,
                                                                         train_gender, random_state)
                    if mix_gender:
                        # model_filename = model_filename + "_mix_gender"
                        fit_kws = {"y": y_train, "covariates": genders_train, "target_idx": None}
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

                    res['pred_loss'].append(model.losses['pred'][-1])
                    res['code_loss'].append(model.losses['code'][-1])

                    # res['n_iter'].append(n_iter)
                    # n_iter += 1
                    res['train_gender'].append(train_gender)
                    res['train_session'].append(train_session)
                    res['split'].append(i_split)
                    res['fold'].append(i_fold)

                    res['lambda'].append(lambda_)
                    # res['test_size'].append(test_size)
                    res['time_used'].append(model.losses['time'][-1])

        res_df = pd.DataFrame.from_dict(res)

        out_filename = 'results_lambda%s_sub_half_%s_%s' % (int(lambda_), run_, random_state)
        if mix_gender:
            out_filename = out_filename + '_mix_gender'
        out_file = os.path.join(out_dir, '%s.csv' % out_filename)
        res_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
