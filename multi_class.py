import os

import numpy as np

import io_
from _base import _pick_half, cross_val


def main():
    # atlas = 'AICHA'
    # data_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    # out_dir = 'D:/ShareFolder/AICHA_VolFC/Result'

    atlas = 'BNA'
    data_dir = 'D:/ShareFolder/BNA/Proc'
    out_dir = 'D:/ShareFolder/BNA/Result'

    sessions = ['REST1', 'REST2']  # session = 'REST1'
    # run = 'LR'
    runs = ['RL', 'LR']
    # connection_type = 'both'  # inter, intra, or both
    connection_type = 'intra'

    info = dict()
    data = dict()

    for session in sessions:
        for run_ in runs:
            info_fname = 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run_)
            info[run_] = io_.read_table(os.path.join(data_dir, info_fname), index_col='ID')
            data[run_] = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

            x, y = _pick_half(data[run_])
            genders = info[run_]["gender"].values

            y_mult = np.zeros(y.shape)
            y_iter = 0
            for sex in (0, 1):
                for label_ in (1, -1):
                    y_ = -1 * np.ones(y.shape)
                    y_[np.where((genders == sex) & (y == label_))] = 1
                    y_mult[np.where((genders == sex) & (y == label_))] = y_iter
                    y_iter += 1
                    res = cross_val(x, y_)
                    print(res)

            res_ = cross_val(x, y_mult)
            print(res_)


if __name__ == '__main__':
    main()
