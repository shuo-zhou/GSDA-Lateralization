import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from pydale.estimator import SIDeRSVM, SIDeRLS


def cross_val(x, y, clf=LinearSVC(), random_state=144, co_variates=None):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
    res = {'acc': [], 'auc': []}
    for train, test in splitter.split(x, y):
        if co_variates is None:
            clf.fit(x[train], y[train])
        else:
            kwargs = {'co_variates': co_variates[train]}
            clf.fit(x[train], y[train], **kwargs)
        y_pred = clf.predict(x[test])
        y_dec = clf.decision_function(x[test])
        res['acc'].append(accuracy_score(y[test], y_pred))
        if len(np.unique(y)) == 2:
            res['auc'].append(roc_auc_score(y[test], y_dec))
        else:
            res['auc'].append(0)

    res['acc_mean'] = np.mean(res['acc'])
    res['auc_mean'] = np.mean(res['auc'])

    return res


def param_search(x, y, clf_='SVC', co_variates=None, reduce_dim=True):
    estimators = {'SVC': LinearSVC, 'Logistic': LogisticRegression,
                  'Ridge': RidgeClassifier, 'SIDeRSVM': SIDeRSVM,
                  'SIDeRLS': SIDeRLS}
    param_grids = {'PCA': {'n_components': [500, 300, 200, 100]},
                   'SVC': {'C': np.logspace(-4, 3, 8)},
                   'Logistic': {'C': np.logspace(-4, 3, 8)},
                   'Ridge': {'eig_vectors': np.logspace(-4, 3, 8)},
                   'SIDeRSVM': {'C': np.logspace(-4, 3, 8),
                                'lambda_': np.logspace(-4, 3, 8)},
                   'SIDeRLS': {'sigma_': np.logspace(-4, 3, 8),
                               'lambda_': np.logspace(-4, 3, 8)},
                   }
    default_params = {'SVC': {'C': 1.0, 'max_iter': 1000000},
                      'Logistic': {'C': 1.0},
                      'Ridge': {'eig_vectors': 1.0},
                      'SIDeRSVM': {'C': 1.0, 'lambda_': 1.0, 'kernel': 'linear'},
                      'SIDeRLS': {'sigma_': 1.0, 'lambda_': 1.0, 'kernel': 'linear'},
                      'PCA': {'n_components': 100}}

    param_grid = param_grids[clf_]
    best_params = {'PCA': default_params['PCA'].copy(),
                   'clf': default_params[clf_].copy()}
    estimator = estimators[clf_]
    kwd_params = best_params['clf'].copy()
    clf_default = estimator(**kwd_params)

    best_estimator = dict()

    if reduce_dim:
        best_acc = 0
        # search for best params for PCA
        for n_comp in param_grids['PCA']['n_components']:
            pca = PCA(n_components=n_comp, random_state=144)
            x_transformed = pca.fit_transform(x)

            res = cross_val(x_transformed, y, clf_default, co_variates=co_variates)
            if res['acc_mean'] > best_acc:
                best_acc = res['acc_mean']
                best_params['PCA']['n_components'] = n_comp

            if best_acc == 1.0:
                best_estimator = {'PCA': pca,
                                  'clf': clf_default}
                return best_estimator, best_params

        pca = PCA(**best_params['PCA'])
        best_estimator['PCA'] = pca
        x_transformed = pca.fit_transform(x)
    else:
        x_transformed = x.copy()

    # search for best params for classifiers
    for param in param_grid:
        kwd_params = {key: best_params['clf'][key]
                      for key in best_params['clf'] if key != param}
        best_acc = 0
        for param_val in param_grid[param]:
            kwd_ = kwd_params.copy()
            kwd_[param] = param_val
            clf = estimator(**kwd_)
            res = cross_val(x_transformed, y, clf, co_variates=co_variates)
            if res['acc_mean'] > best_acc:
                best_acc = res['acc_mean']
                best_params['clf'][param] = param_val

            if best_acc == 1.0:
                best_estimator['clf'] = clf
                return best_estimator, best_params

    best_estimator['clf'] = estimator(**best_params['clf'])

    return best_estimator, best_params


def _pick_half(data, random_state=144):
    x = np.zeros(data['Left'].shape)
    left_idx, right_idx = train_test_split(range(x.shape[0]), test_size=0.5,
                                           random_state=random_state)
    x[left_idx] = data['Left'][left_idx]
    x[right_idx] = data['Right'][right_idx]

    n_sub = x.shape[0]
    y = np.zeros(n_sub)
    y[left_idx] = 1
    y[right_idx] = -1

    return x, y
