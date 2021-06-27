import os
# import pickle
import io_
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
# from sklearn.metrics import accuracy_score, roc_auc_score
from _base import _pick_half
import pandas as pd
from pydale.estimator import CoDeLR
from torchmetrics.functional import accuracy


def main():
    atlas = 'BNA'
    data_dir = "/media/shuoz/MyDrive/HCP/%s/Proc" % atlas
    out_dir = '/media/shuoz/MyDrive/HCP/%s/Results' % atlas
    # data_dir = 'D:/ShareFolder/BNA/Proc'
    # out_dir = 'D:/ShareFolder/BNA/Result'
    # atlas = 'AICHA'
    # data_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    # out_dir = 'D:/ShareFolder/AICHA_VolFC/Result'
    session = 'REST1'  # session = 'REST1'
    run_ = 'LR'
    # runs = ['RL', 'LR']
    connection_type = 'intra'
    random_state = 144
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    l2_param = 0.1
    test_sizes = [0.1, 0.2, 0.3, 0.4]

    info = dict()
    data = dict()

    info_file = 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run_)
    info[run_] = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')
    data[run_] = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

    x, y = _pick_half(data[run_], random_state=random_state)
    y = label_binarize(y, classes=[-1, 1]).reshape(-1)

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    genders = info[run_]['gender'].values
    idx_male = np.where(genders == 0)[0]
    idx_female = np.where(genders == 1)[0]

    x = torch.from_numpy(x)
    x = x.float()
    y = torch.from_numpy(y)
    y = y.long()
    genders = torch.from_numpy(genders.reshape((-1, 1)))
    genders = genders.float()
    res = {"acc_within": [], "acc_without": [], 'pred_loss': [], 'code_loss': [], 'lambda': [], 'test_size': [],
           'time_used': [], 'n_iter': []}
    for test_size in test_sizes:
        spliter = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)

        for lambda_ in lambdas:
            for train_idx, test_idx in [(idx_male, idx_female), (idx_female, idx_male)]:
                n_iter = 0
                for train, test in spliter.split(train_idx, y[train_idx]):
                    train_ = train_idx[train]
                    test_ = train_idx[test]
                    model = CoDeLR(lambda_=lambda_, l2_hparam=l2_param)
                    model.fit(x, y[train_], genders, train_idx=train_)
                    y_pred_wi = model.predict(x[test_])
                    acc_wi = accuracy(y[test_], y_pred_wi)
                    res['acc_within'].append(acc_wi.item())

                    y_pred_wo = model.predict(x[test_idx])
                    acc_wo = accuracy(y[test_idx], y_pred_wo)
                    res['acc_without'].append(acc_wo.item())

                    out = model.forward(x)
                    pred_loss = model._compute_pred_loss(out[train], y[train])
                    res['pred_loss'].append(pred_loss.item())

                    code_loss = model._compute_code_loss(out, genders)
                    res['code_loss'].append(code_loss.item())

                    model_path = os.path.join(out_dir, "lambda_%s_test_%s_iter_%s.pt" % (lambda_, test_size, n_iter))
                    torch.save(model, model_path)
                    res['n_iter'].append(n_iter)
                    n_iter += 1

                    res['lambda'].append(lambda_)
                    res['test_size'].append(test_size)
                    res['time_used'].append(model.losses['time'][-1])

    res_df = pd.DataFrame.from_dict(res)
    out_file = os.path.join(out_dir, 'results.csv')
    res_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
