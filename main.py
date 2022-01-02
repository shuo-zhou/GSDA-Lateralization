import os
import pickle
import io_
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from _base import cross_val, param_search, _pick_half


def main():
    data_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    # data_dir = 'D:/ShareFolder/BNA/Proc'
    session = 'REST2'  # session = 'REST1'
    # run = 'LR'
    runs = ['RL', 'LR']
    # connection_type = 'both'  # inter, intra, or both
    connection_type = 'intra'
    random_state = 144
    clf = 'SVC'
    # clf = 'SIDeRSVM'
    atlas = 'AICHA'
    # out_dir = 'D:/ShareFolder/BNA/Result'
    out_dir = 'D:/ShareFolder/AICHA_VolFC/Result'

    # info_fname = 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run)
    # info = io_.read_table(os.path.join(data_dir, info_fname), index_col=None)

    # male_info = info.loc[info['gender'] == 0]
    # male_idx = male_info.index
    #
    # female_info = info.loc[info['gender'] == 1]
    # female_idx = female_info.index

    info = dict()
    data = dict()

    for run_ in runs:
        info_fname = 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run_)
        info[run_] = io_.read_table(os.path.join(data_dir, info_fname), index_col='ID')
        data[run_] = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)
    x = data['RL']['Left']
    y = info['RL']['gender'].values
    res = cross_val(x, y)

    x = dict()
    y = dict()
    genders = dict()
    for i in range(2):
        # find the overlap subjects between RL and LR runs
        idx = info[runs[i]].index.isin(info[runs[1 - i]].index)
        for side in ['Left', 'Right']:
            data[runs[i]][side] = data[runs[i]][side][idx]
        genders[runs[i]] = info[runs[i]].loc[idx, 'gender'].values
        # Data
        x[runs[i]], y[runs[i]] = _pick_half(data[runs[i]], random_state=random_state)

    # y['LR'] and y['RL'] should be equal here
    if np.array_equal(y['LR'], y['RL']) and np.array_equal(genders['LR'], genders['RL']):
        y = np.copy(y['LR'])
        genders = genders['LR'].reshape((-1, 1))
    else:
        raise ValueError('Labels and gender information across runs do not match')

    # data_fname = 'HCP_%s_half_brain_%s_%s.hdf5' % (connection_type, session, run)
    # data = io_.load_hdf5(os.path.join(data_dir, data_fname))
    # data = io_.load_half_brain(data_dir, session, run, connection_type)

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                                      random_state=random_state)
    splits = splitter.split(x['LR'], y)

    res = {'acc': [], 'auc': [], 'embed': [], 'clf': [], 'params': []}

    for train, test in splits:
        # Normalisation
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x['LR'][train])

        # search for best params
        if clf not in ['SIDeRSVM', 'SIDeRLS']:
            co_variates = None
        else:
            co_variates = genders[train]
        best_estimator, best_params = param_search(x_train, y[train], clf_=clf,
                                                   co_variates=co_variates)

        # Training
        x_train_pc = best_estimator['PCA'].fit_transform(x_train)
        if clf in ['SIDeRSVM', 'SIDeRLS']:
            best_estimator['clf'].fit(x_train_pc, y[train], )
        else:
            best_estimator['clf'].fit(x_train_pc, y[train], )
        res['embed'].append(best_estimator['PCA'])
        res['clf'].append(best_estimator['clf'])
        res['params'].append(best_params)

        # Testing
        x_test = scaler.transform(x['RL'][test])
        x_test_pc = best_estimator['PCA'].transform(x_test)
        y_pred = best_estimator['clf'].predict(x_test_pc)
        y_dec = best_estimator['clf'].decision_function(x_test_pc)

        res['acc'].append(accuracy_score(y[test], y_pred))
        res['auc'].append(roc_auc_score(y[test], y_dec))

    # out_fname = '%s_half_brain_%s_%s_%s.pickle' % \
    #             (connection_type, session, run, random_state)
    out_fname = '%s_half_brain_%s_%s_%s.pickle' % (connection_type, clf, session, random_state)
    outfile = open(os.path.join(out_dir, out_fname), 'wb')
    pickle.dump(res, outfile)
    outfile.close()


if __name__ == '__main__':
    main()
