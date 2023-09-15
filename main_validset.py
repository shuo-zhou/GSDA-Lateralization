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
    dataset = cfg.DATASET.DATASET

    random_state = cfg.SOLVER.SEED
    connection_type = cfg.DATASET.CONNECTION
    lambda_ = cfg.SOLVER.LAMBDA_
    l2_param = cfg.SOLVER.L2PARAM
    num_repeat = cfg.DATASET.NUM_REPEAT
    run_ = cfg.DATASET.RUN
    test_size = cfg.DATASET.TEST_SIZE
    mix_gender = cfg.DATASET.MIX_GEND
    if mix_gender:
        lambda_ = 0.0
        out_folder = "lambda0_mix_gender"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # test_sizes = [0.1, 0.2, 0.3, 0.4]

    # info_file = 'HCP_%s_half_brain_gender_equal.csv' % atlas
    info_file = '%s_%s_half_brain.csv' % (cfg.DATASET.DATASET, atlas)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')

    data = io_.load_half_brain(data_dir, atlas, session=None, run=run_,
                               connection_type=connection_type, dataset=dataset)
    groups = info['gender'].values

    # main loop
    res = {"acc_ic": [], "acc_oc": [], 'pred_loss': [], 'code_loss': [], 'lambda': [],
           'split': [], 'fold': [], 'train_gender': [], 'time_used': []}
    if 0 < test_size < 1:
        res['acc_ic_test_sub'] = []
        res['acc_oc_test_sub'] = []

    # for test_size in test_sizes:
    #     spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)

    for i_split in range(num_repeat):

        x, y, x1, y1 = _pick_half(data, random_state=random_state * (i_split + 1))
        y = label_binarize(y, classes=[-1, 1]).reshape(-1)
        y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

        x_all = [x, x1]
        y_all = [y, y1]

        if 0 < test_size < 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state * (i_split + 1))
            train_sub, test_sub = sss.split(x, groups)

        for i_fold in [0, 1]:
            x_train = x_all[i_fold]
            y_train = y_all[i_fold]
            x_test_ = x_all[1 - i_fold]
            y_test_ = y_all[1 - i_fold]
            # scaler = StandardScaler()
            # scaler.fit(x_train)
            for train_group in [0, 1]:
                ic_idx = np.where(groups == train_group)[0]
                oc_idx = np.where(groups == 1 - train_group)[0]

                if mix_gender:
                    train_group = "mix"
                    if train_group == 1:
                        continue

                x_test_ic = x_test_[ic_idx]
                y_test_ic = y_test_[ic_idx]

                x_test_oc = x_test_[oc_idx]
                y_test_oc = y_test_[oc_idx]

                xy_test = {"acc_ic": [x_test_ic, y_test_ic], "acc_oc": [x_test_oc, y_test_oc]}

                model = GSLR(lambda_=lambda_, C=l2_param, max_iter=5000)
                fit_kws = {"y": y_train[ic_idx], "groups": groups, "target_idx": ic_idx}
                model_filename = "%s_lambda%s_%s_%s_gender_%s_%s" % (dataset, int(lambda_), i_split, i_fold,
                                                                     train_group, random_state)
                if mix_gender:
                    model_filename = model_filename + "_mix_gender"
                    fit_kws = {"y": y_train, "groups": groups, "target_idx": None}
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
                res['train_gender'].append(train_group)
                res['split'].append(i_split)
                res['fold'].append(i_fold)

                res['lambda'].append(lambda_)
                # res['test_size'].append(test_size)
                res['time_used'].append(model.losses['time'][-1])

        res_df = pd.DataFrame.from_dict(res)

        out_filename = 'results_%s_lambda%s_sub_half_%s_%s' % (dataset, int(lambda_), run_, random_state)
        if mix_gender:
            out_filename = out_filename + '_mix_gender'
        out_file = os.path.join(out_dir, '%s.csv' % out_filename)
        res_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
