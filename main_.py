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
    data_dir = "/media/shuo/MyDrive/data/HCP/%s/Proc" % atlas
    out_dir = '/media/shuo/MyDrive/data/HCP/%s/Results/' % atlas
    # data_dir = 'D:/ShareFolder/BNA/Proc'
    # out_dir = 'D:/ShareFolder/BNA/Result'
    # atlas = 'AICHA'
    # data_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    # out_dir = 'D:/ShareFolder/AICHA_VolFC/Result'
    sessions = ['REST1', 'REST2']  # session = 'REST1'
    run_ = 'Fisherz'
    # runs = ['RL', 'LR']
    connection_type = 'intra'
    random_state = 144
    l2_param = 0.1

    x_all = dict()
    y_all = dict()
    genders = dict()

    info_file = 'HCP_%s_half_brain.csv' % atlas
    info = io_.read_table(os.path.join(data_dir, info_file), index_col='ID')
    for session in sessions:
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
        gender = info['gender'].values
        gender = torch.from_numpy(gender.reshape((-1, 1)))

        genders[session] = torch.cat((gender, gender))

    for i in range(len(sessions)):
        for tgt_sex in [0, 1]:
            model_path = os.path.join(out_dir, "model_%s_%s.pt" % (sessions[i], tgt_sex))
            tgt_idx = torch.where(genders[sessions[i]][:, 0] == tgt_sex)
            src_idx = torch.where(genders[sessions[i]][:, 0] != tgt_sex)
            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                model = CoDeLR(lambda_=2.0, l2_hparam=l2_param)
                model.fit(x_all[sessions[i]], y_all[sessions[i]][tgt_idx], genders[sessions[i]], target_idx=tgt_idx)
                torch.save(model, model_path)

            y_pred = model.predict(x_all[sessions[1-i]])
            print(accuracy(y_all[sessions[1 - i]][tgt_idx], y_pred[tgt_idx].view(-1).int()))
            print(accuracy(y_all[sessions[1 - i]][src_idx], y_pred[src_idx].view(-1).int()))


if __name__ == '__main__':
    main()
