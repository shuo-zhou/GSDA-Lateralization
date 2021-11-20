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
    out_dir = '/media/shuoz/MyDrive/HCP/%s/Results/Rand_Half_Fisherz/' % atlas
    # data_dir = 'D:/ShareFolder/BNA/Proc'
    # out_dir = 'D:/ShareFolder/BNA/Result'
    # atlas = 'AICHA'
    # data_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    # out_dir = 'D:/ShareFolder/AICHA_VolFC/Result'
    sessions = ['REST1', 'REST2']  # session = 'REST1'
    run_ = 'Fisherz'
    connection_type = 'intra'
    random_state = 144
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    l2_param = 0.1
    test_sizes = [0.1, 0.2, 0.3, 0.4]

    info = dict()
    x_all = dict()
    y_all = dict()
    genders = dict()

    np.random.seed(144)

    for session in sessions:
        info_file = 'HCP_%s_half_brain_%s.csv' % (atlas, session)
        info[session] = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')
        data = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

        x, y, x1, y1 = _pick_half(data, random_state=random_state)
        # x, y = _pick_half_subs(data[run_])
        y = label_binarize(y, classes=[-1, 1]).reshape(-1)
        y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

        x = torch.from_numpy(x)
        x = x.float()
        x1 = torch.from_numpy(x1)
        x1 = x1.float()

        y = torch.from_numpy(y)
        y = y.long()
        y1 = torch.from_numpy(y1)
        y1 = y1.long()

        x_all[session] = torch.cat([x, x1])
        y_all[session] = torch.cat([y, y1])
        gender = info[session]['gender'].values
        gender = torch.from_numpy(gender.reshape((-1, 1)))

        genders[session] = torch.cat((gender, gender))

    res = {"acc_ic_is": [], "acc_ic_os": [], "acc_oc_is": [], "acc_oc_os": [], 'pred_loss': [],
           'code_loss': [], 'lambda': [], 'time_used': [], 'train_session': [], 'split': [], 'train_gender': []}

    for i in range(1000):
        lambda_ = 2.0

        train_ses_ = np.random.randint(2)
        train_session = sessions[train_ses_]
        test_session = sessions[1 - train_ses_]
        x_train_ = x_all[train_session]
        y_train_ = y_all[train_session]
        g_train_ = genders[train_session]
        x_test_ = x_all[test_session]
        y_test_ = y_all[test_session]
        g_test_ = genders[test_session]
        spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

        train, test = next(spliter.split(x_train_, y_train_))

        for train_gender in [0, 1]:
            ic_is_train = np.where(g_train_[train] == train_gender)[0]
            oc_is_train = np.where(g_train_[train] == 1 - train_gender)[0]

            ic_is_test = np.where(g_train_[test] == train_gender)[0]
            oc_is_test = np.where(g_train_[test] == 1 - train_gender)[0]

            ic_os_idx = np.where(g_test_ == train_gender)[0]
            oc_os_idx = np.where(g_test_ == 1 - train_gender)[0]

            x_test_ic_is = x_train_[test][ic_is_test]
            y_test_ic_is = y_train_[test][ic_is_test]
            x_test_oc_is = x_train_[test][oc_is_test]
            y_test_oc_is = y_train_[test][oc_is_test]
            x_test_ic_os = x_test_[ic_os_idx]
            y_test_ic_os = y_test_[ic_os_idx]

            x_test_oc_os = x_test_[oc_os_idx]
            y_test_oc_os = y_test_[oc_os_idx]

            xy_test = {"acc_ic_is": [x_test_ic_is, y_test_ic_is], "acc_ic_os": [x_test_ic_os, y_test_ic_os],
                       "acc_oc_is": [x_test_oc_is, y_test_oc_is], "acc_oc_os": [x_test_oc_os, y_test_oc_os]}

            model_path = os.path.join(out_dir, "rand_half_permute_lambda_%s_%s_%s_gender_%s.pt" %
                                      (lambda_, train_session, i, train_gender))

            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                x_train = x_train_[train]
                y_train = y_train_[train][ic_is_train]
                model = CoDeLR(lambda_=lambda_, l2_hparam=l2_param)
                model.fit(x_train, y_train[torch.randperm(y_train.shape[0])], g_train_[train], target_idx=ic_is_train)
                torch.save(model, model_path)

            for acc_key in xy_test:
                test_x, test_y = xy_test[acc_key]
                y_pred_ = model.predict(test_x)
                acc_ = accuracy(test_y, y_pred_.view(-1).int())
                res[acc_key].append(acc_.item())

            res['pred_loss'].append(model.losses['pred'][-1])
            res['code_loss'].append(model.losses['code'][-1])

            res['train_gender'].append(train_gender)
            res['train_session'].append(train_session)
            res['split'].append(i)

            res['lambda'].append(lambda_)
            # res['test_size'].append(test_size)
            res['time_used'].append(model.losses['time'][-1])

    res_df = pd.DataFrame.from_dict(res)
    out_file = os.path.join(out_dir, 'results_random_half_split.csv')
    res_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
