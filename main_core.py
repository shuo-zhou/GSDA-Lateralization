import os
# import pickle
import io_
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
# from sklearn.metrics import accuracy_score, roc_auc_score
from _base import _pick_half, _pick_half_subs
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
    # sessions = ['REST1', 'REST2']
    session = 'REST1'
    run_ = 'Fisherz'
    # runs = ['RL', 'LR']
    connection_type = 'intra'
    random_state = 144
    # lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    lambdas = [1.0]
    l2_param = 0.1
    # test_sizes = [0.1, 0.2, 0.3, 0.4]
    test_sizes = [0.5]

    # for session in sessions:
    info_file = 'HCP_%s_half_brain_%s.csv' % (atlas, session)
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')
    data = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

    x, y, x1, y1 = _pick_half(data, random_state=random_state)

    y = label_binarize(y, classes=[-1, 1]).reshape(-1)
    y = torch.from_numpy(y)
    y = y.long()
    y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)
    y1 = torch.from_numpy(y1)
    y1 = y1.long()

    x = torch.from_numpy(x)
    x = x.float()
    x1 = torch.from_numpy(x1)
    x1 = x1.float()

    genders = info['gender'].values
    idx_male = np.where(genders == 0)[0]
    idx_female = np.where(genders == 1)[0]
    genders = torch.from_numpy(genders.reshape((-1, 1)))
    genders = genders.float()
    res = {"acc_within_new_sub": [], "acc_within_same_sub": [], "acc_without_trained": [], "acc_without_untrained": [],
           'pred_loss': [], 'code_loss': [], 'lambda': [], 'test_size': [], 'time_used': [], 'n_iter': []}
    for test_size in test_sizes:
        spliter = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=random_state)

        for lambda_ in lambdas:
            for train_idx, test_idx in [(idx_male, idx_female), (idx_female, idx_male)]:
                n_iter = 0
                for train, test in spliter.split(train_idx, y[train_idx]):
                    train_ = train_idx[train]
                    test_ = train_idx[test]

                    model = CoDeLR(lambda_=lambda_, l2_hparam=l2_param)
                    # x_train = torch.cat((x[train_], x[test_idx]))
                    # genders_train = torch.cat((genders[train_], genders[test_idx]))
                    # y_train = y[train_]
                    # model.fit(x_train, y_train, genders_train, train_idx=np.arange(len(train_)))
                    x_train = torch.cat((x[train_], x1[train_], x[test_idx]))
                    genders_train = torch.cat((genders[train_], genders[train_], genders[test_idx]))
                    y_train = torch.cat((y[train_], y[train_]))
                    model.fit(x_train, y_train, genders_train, target_idx=np.arange(len(train_) * 2))

                    y_pred_wi = model.predict(torch.cat((x[test_], x1[test_])))
                    acc_wi = accuracy(torch.cat((y[test_], y1[test_])), y_pred_wi)
                    res['acc_within_new_sub'].append(acc_wi.item())

                    y_pred_wi0 = model.predict(x1[train_])
                    acc_wo = accuracy(y1[train_], y_pred_wi0)
                    res['acc_within_same_sub'].append(acc_wo.item())

                    y_pred_wo = model.predict(x[test_idx])
                    acc_wo = accuracy(y[test_idx], y_pred_wo)
                    res['acc_without_trained'].append(acc_wo.item())

                    y_pred_wo0 = model.predict(x1[test_idx])
                    acc_wo0 = accuracy(y1[test_idx], y_pred_wo0)
                    res['acc_without_untrained'].append(acc_wo0.item())

                    out = model.forward(x)
                    pred_loss = model._compute_pred_loss(out[train], y[train])
                    res['pred_loss'].append(pred_loss.item())

                    code_loss = model._compute_code_loss(out, genders)
                    res['code_loss'].append(code_loss.item())

                    model_path = os.path.join(out_dir, "lambda_%s_test_%s1_iter_%s.pt" % (lambda_, test_size, n_iter))
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
