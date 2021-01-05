import os
import pickle
import io_
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score


def cross_val(X, y, clf, random_state=144):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                                      random_state=random_state)
    res = {'acc': [], 'auc': []}
    for train, test in splitter.split(X, y):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        y_dec = clf.decision_function(X[test])
        res['acc'].append(accuracy_score(y[test], y_pred))
        res['auc'].append(roc_auc_score(y[test], y_dec))

    res['acc_mean'] = np.mean(res['acc'])
    res['auc_mean'] = np.mean(res['auc'])

    return res


def param_search(X, y, clf_='SVC'):
    estimators = {'SVC': SVC}
    param_grids = {'PCA': {'n_components': [500, 300, 200, 100]},
                   'SVC': {'C': np.logspace(-4, 3, 8)}}
    default_params = {'SVC': {'C': 1, 'kernel': 'linear'},
                      'PCA': {'n_components': 100}}

    param_grid = param_grids[clf_]
    best_params = {'PCA': default_params['PCA'].copy(),
                   'clf': default_params[clf_].copy()}
    estimator = estimators[clf_]
    kwd_params = best_params['clf'].copy()
    clf_default = estimator(**kwd_params)

    best_acc = 0
    # search for best params for PCA
    for n_comp in param_grids['PCA']['n_components']:
        pca = PCA(n_components=n_comp, random_state=144)
        x_pc = pca.fit_transform(X)

        res = cross_val(x_pc, y, clf_default)
        if res['acc_mean'] > best_acc:
            best_acc = res['acc_mean']
            best_params['PCA']['n_components'] = n_comp

        if best_acc == 1.0:
            best_estimator = {'PCA': pca,
                              'clf': clf_default}
            return best_estimator, best_params

    pca = PCA(**best_params['PCA'])
    x_pc = pca.fit_transform(X)

    # search for best params for classifiers
    for param in param_grid:
        kwd_params = {key: best_params['clf'][key]
                      for key in best_params['clf'] if key != param}
        best_acc = 0
        for param_val in param_grid[param]:
            kwd_ = kwd_params.copy()
            kwd_[param] = param_val
            clf = estimator(**kwd_)
            res = cross_val(x_pc, y, clf)
            if res['acc_mean'] > best_acc:
                best_acc = res['acc_mean']
                best_params['clf'][param] = param_val

            if best_acc == 1.0:
                best_estimator = {'PCA': pca,
                                  'clf': clf}
                return best_estimator, best_params

    best_estimator = {'PCA': pca,
                      'clf': estimator(**best_params['clf'])}

    return best_estimator, best_params


def main():
    datadir = 'D:/ShareFolder/AICHA_VolFC/Proc'
    session = 'REST1'
    run = 'LR'
    connection_type = 'intra'
    random_state = 144

    output_dir = 'D:/ShareFolder/AICHA_VolFC/Result'

    info_fname = 'HCP_half_brain_%s_%s.csv' % (session, run)
    info = io_.read_table(os.path.join(datadir, info_fname), index_col=None)

    data_fname = 'HCP_%s_half_brain_%s_%s.hdf5' % (connection_type, session, run)
    data = io_.load_hdf5(os.path.join(datadir, data_fname))

    # split = ShuffleSplit(train_size=0.8, test_size=0.2)

    # idx = {'left': {'male': dict()},
    #        'right': {'female': dict()}}

    male_info = info.loc[info['gender'] == 0]
    male_idx = male_info.index

    female_info = info.loc[info['gender'] == 1]
    female_idx = female_info.index

    left_idx, right_idx = train_test_split(info.index,
                                           test_size=0.5,
                                           random_state=random_state)

    # Data
    X = np.zeros(data['Left'].shape)
    X[left_idx] = data['Left'][left_idx]
    X[right_idx] = data['Right'][right_idx]
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Label
    n_sub = X.shape[0]
    y = np.zeros(n_sub)
    y[left_idx] = 1
    y[right_idx] = -1

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

    res = {'acc': [], 'auc': [], 'embed': [], 'clf': [], 'params': []}

    for train, test in splitter.split(X, y):
        # Normalisation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        # search for best params
        best_estimator, best_params = param_search(X_train, y[train])
        # Training
        X_train_pc = best_estimator['PCA'].fit_transform(X_train)
        best_estimator['clf'].fit(X_train_pc, y[train])
        res['embed'].append(best_estimator['PCA'])
        res['clf'].append(best_estimator['clf'])
        res['params'].append(best_params)
        # Testing
        X_test = scaler.transform(X[test])
        X_test_pc = best_estimator['PCA'].transform(X_test)
        y_pred = best_estimator['clf'].predict(X_test_pc)
        y_dec = best_estimator['clf'].decision_function(X_test_pc)

        res['acc'].append(accuracy_score(y[test], y_pred))
        res['auc'].append(roc_auc_score(y[test], y_dec))

    out_fname = '%s_half_brain_%s_%s_%s.pickle' % \
                (connection_type, session, run, random_state)
    outfile = open(os.path.join(output_dir, out_fname), 'wb')
    pickle.dump(res, outfile)
    outfile.close()


if __name__ == '__main__':
    main()
