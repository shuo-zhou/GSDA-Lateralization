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
    out_dir = cfg.OUTPUT.ROOT
    run_ = cfg.DATASET.RUN

    random_state = cfg.SOLVER.SEED
    lambda_ = cfg.SOLVER.LAMBDA_
    l2_param = cfg.SOLVER.L2PARAM
    num_repeat = cfg.DATASET.NUM_REPEAT
    test_sizes = [0.1, 0.2, 0.3, 0.4]

    # info_file = 'HCP_%s_half_brain_gender_equal.csv' % atlas
    info_file = 'Phenotype_filt_noqc.csv'
    info = io_.read_table(os.path.join(data_dir, info_file))

    # dx_label = info['DX_GROUP'].values
    # health_idx = np.where(dx_label == 1)[0]
    male_idx = np.where(info['SEX'].values == 1)[0]
    sites = info["SITE_ID"].values
    sites = sites[male_idx]
    # covariates = info['SEX'].values.reshape((-1, 1))
    covariates = info['DX_GROUP'].values
    # covariates[covariates == 1] = 0
    # covariates[covariates == 2] = 1
    nyu_idx = np.where(sites == "NYU")[0]
    covariates = covariates[male_idx][nyu_idx]
    covariates[covariates == 2] = 0
    covariates = covariates.reshape((-1, 1))
    data = io_.load_half_brain(data_dir, atlas, data_type=cfg.DATASET.TYPE, dataset=cfg.DATASET.DATASET)
    data["Left"] = data["Left"][male_idx][nyu_idx]
    data["Right"] = data["Right"][male_idx][nyu_idx]

    res = {"acc_ic": [], "acc_oc": [], 'pred_loss': [], 'code_loss': [], 'lambda': [], 'split': [], 'fold': [],
           'train_covariate': [], 'time_used': []}
    # for test_size in test_sizes:
    #     spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)

    for i_split in range(num_repeat):
        x, y, x1, y1 = _pick_half(data, random_state=random_state * (i_split + 1))
        # x, y = _pick_half_subs(data[run_])
        y = label_binarize(y, classes=[-1, 1]).reshape(-1)
        y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

        x_all = [x, x1]
        y_all = [y, y1]

        for i_fold in [0, 1]:
            x_train = x_all[i_fold]
            y_train = y_all[i_fold]
            x_test_ = x_all[1 - i_fold]
            y_test_ = y_all[1 - i_fold]
            # scaler = StandardScaler()
            # scaler.fit(x_train)
            for train_covariate in [0, 1]:
                ic_idx = np.where(covariates == train_covariate)[0]
                oc_idx = np.where(covariates == 1 - train_covariate)[0]

                x_test_ic = x_test_[ic_idx]
                y_test_ic = y_test_[ic_idx]

                x_test_oc = x_test_[oc_idx]
                y_test_oc = y_test_[oc_idx]

                xy_test = {"acc_ic": [x_test_ic, y_test_ic], "acc_oc": [x_test_oc, y_test_oc]}

                model = CoDeLR(lambda_=lambda_, C=l2_param, max_iter=5000)
                model_path = os.path.join(out_dir, "abide_lambda_%s_%s_%s_gender_%s_%s.pt" %
                                          (lambda_, i_split, i_fold, train_covariate, random_state))
                if os.path.exists(model_path):
                    model = torch.load(model_path)
                else:
                    model.fit(x_train, y_train[ic_idx], covariates, target_idx=ic_idx)
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
                res['train_covariate'].append(train_covariate)
                res['split'].append(i_split)
                res['fold'].append(i_fold)

                res['lambda'].append(lambda_)
                # res['test_size'].append(test_size)
                res['time_used'].append(model.losses['time'][-1])

    res_df = pd.DataFrame.from_dict(res)
    out_file = os.path.join(out_dir, 'abide_results_lambda_%s_sub_half_%s_%s.csv' % (lambda_, run_, random_state))
    res_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
