# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
from numpy.linalg import multi_dot
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from ._base import _BaseTransformer


class MIDA(_BaseTransformer):
    def __init__(
        self,
        n_components,
        kernel="linear",
        lambda_=1.0,
        mu=1.0,
        eta=1.0,
        aug=True,
        n_jobs=1.0,
        alpha=1.0,
        kernel_params=None,
        fit_inverse_transform=False,
        **kwargs
    ):
        """Maximum independence domain adaptation

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= d)
        kernel : str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        mu: total captured variance param
        lambda_: label dependence param

        References
        ----------
        Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace
        using domain features and independence maximization. IEEE transactions on
        cybernetics, 48(1), pp.288-299.
        """
        super().__init__(kernel, n_jobs, alpha, kernel_params, fit_inverse_transform)
        self.n_components = n_components
        # self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.eta = eta
        # self.n_jobs = n_jobs
        self.aug = aug
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)
        # self.kernel_params = kernel_params
        # self.alpha = alpha
        # self.fit_inverse_transform = fit_inverse_transform
        self.kwargs = kwargs

    def fit(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Labels, shape (nl_samples,)
        co_variates : array-like
            Domain co-variates, shape (n_samples, n_co-variates)

        Note
        ----
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        """
        if self.aug and type(co_variates) == np.ndarray:
            X = np.concatenate((X, co_variates), axis=1)
        ker_x = self._get_kernel(X)
        n_samples = X.shape[0]
        unit_mat, ctr_mat = self._get_unit_ctr_mat(n_samples)
        if type(co_variates) == np.ndarray:
            ker_c = np.dot(co_variates, co_variates.T)
        else:
            ker_c = np.zeros((n_samples, n_samples))
        obj_min = (
            multi_dot([ker_x, ctr_mat, ker_c, ctr_mat, ker_x])
            / np.square(n_samples - 1)
            + self.eta * unit_mat
        )
        if y is not None:
            y_mat = self._lb.fit_transform(y)
            ker_y = np.dot(y_mat, y_mat.T)
            obj_max = multi_dot(
                [
                    ker_x,
                    self.mu * ctr_mat
                    + self.lambda_ * multi_dot([ctr_mat, ker_y, ctr_mat]),
                    ker_x,
                ]
            )
        # obj_min = np.trace(np.dot(K,L))
        else:
            obj_max = self.mu * multi_dot([ker_x, ctr_mat, ker_x])

        # objective = multi_dot([ker_x, (obj_max - obj_min), ker_x.T])
        objective = np.dot(linalg.inv(obj_min), obj_max)
        eig_values, eig_vectors = linalg.eigh(objective)
        idx_sorted = (-1 * eig_values).argsort()

        self.eig_vectors = eig_vectors[:, idx_sorted][:, : self.n_components]
        self.eig_vectors = np.asarray(self.eig_vectors, dtype=np.float)
        self.eig_values = eig_values[idx_sorted][: self.n_components]

        if self.fit_inverse_transform:
            scaled_eig_vec = self.eig_vectors / np.sqrt(np.abs(self.eig_values))
            X_transformed = np.dot(ker_x, scaled_eig_vec)
            self._fit_inverse_transform(X_transformed, X)

        self.X_fit = X
        return self

    def transform(self, X, co_variates=None):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)
        co_variates : array-like,
            Domain co-variates, shape (n_samples, n_co-variates)
        Returns
        -------
        array-like
            transformed data
        """
        check_is_fitted(self, "X_fit")
        if self.aug and type(co_variates) == np.ndarray:
            X = np.concatenate((X, co_variates), axis=1)
        ker_x = self._get_kernel(X, self.X_fit)
        scaled_eig_vec = self.eig_vectors / np.sqrt(np.abs(self.eig_values))

        return np.dot(ker_x, scaled_eig_vec)
        # return np.dot(ker_x, self.eig_vectors)

    def fit_transform(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)
        y : array-like
            shape (n_samples,)
        co_variates : array-like
            shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            transformed X_transformed
        """
        self.fit(X, y, co_variates)

        return self.transform(X)


class LinearMIDA(_BaseTransformer):
    def __init__(
        self,
        n_components,
        lambda_=1.0,
        mu=1.0,
        eta=1.0,
        aug=False,
        n_jobs=1.0,
        **kwargs
    ):
        """Maximum independence domain adaptation

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= d)
        kernel : str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        mu: total captured variance param
        lambda_: label dependence param

        """
        super().__init__(n_jobs)
        self.n_components = n_components
        # self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.eta = eta
        # self.n_jobs = n_jobs
        self.aug = aug
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)
        self.mean_ = 0
        # self.kernel_params = kernel_params
        # self.alpha = alpha
        # self.fit_inverse_transform = fit_inverse_transform
        self.kwargs = kwargs

    def fit(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Labels, shape (nl_samples,)
        co_variates : array-like
            Domain co-variates, shape (n_samples, n_co-variates)

        Note
        ----
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        """
        if self.aug and type(co_variates) == np.ndarray:
            X = np.concatenate((X, co_variates), axis=1)
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_
        n_samples = X.shape[0]
        n_features = X.shape[1]
        unit_mat, ctr_mat = self._get_unit_ctr_mat(n_samples)
        unit_mat = np.eye(n_features)
        if type(co_variates) == np.ndarray:
            ker_c = np.dot(co_variates, co_variates.T)
        else:
            ker_c = np.zeros((n_samples, n_samples))
        obj_min = multi_dot([X.T, ctr_mat, ker_c, ctr_mat, X]) + self.eta * unit_mat
        if y is not None:
            y_mat = self._lb.fit_transform(y)
            ker_y = np.dot(y_mat, y_mat.T)
            obj_max = multi_dot(
                [
                    X.T,
                    self.mu * ctr_mat
                    + self.lambda_ * multi_dot([ctr_mat, ker_y, ctr_mat]),
                    X,
                ]
            )
        # obj_min = np.trace(np.dot(K,L))
        else:
            # obj_max = self.mu * multi_dot([X.T, X])
            # obj_max = self.mu * unit_mat
            obj_max = multi_dot([X.T, X])

        # objective = multi_dot([ker_x, (obj_max - obj_min), ker_x.T])
        objective = np.dot(linalg.inv(obj_min), obj_max)
        # objective = obj_max - obj_min
        eig_values, eig_vectors = eigsh(objective, k=self.n_components)
        idx_sorted = (-1 * eig_values).argsort()

        self.eig_vectors = eig_vectors[:, idx_sorted]
        self.eig_vectors = np.asarray(self.eig_vectors, dtype=np.float)
        self.eig_values = eig_values[idx_sorted]

        return self

    def transform(self, X, co_variates=None):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)
        co_variates : array-like,
            Domain co-variates, shape (n_samples, n_co-variates)
        Returns
        -------
        array-like
            transformed data
        """
        # check_is_fitted(self, 'X_fit')
        if self.aug and type(co_variates) == np.ndarray:
            X = np.concatenate((X, co_variates), axis=1)
        X = X - self.mean_

        return np.dot(X, self.eig_vectors)

    def fit_transform(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)
        y : array-like
            shape (n_samples,)
        co_variates : array-like
            shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            transformed X_transformed
        """
        self.fit(X, y, co_variates)

        return self.transform(X)

    def inverse_transform(self, x):

        x_rec = np.dot(x, self.eig_vectors.T)

        return x_rec + self.mean_
