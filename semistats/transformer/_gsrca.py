# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
from numpy.linalg import multi_dot
from scipy import linalg
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from ._base import _BaseTransformer


class GSRCA(_BaseTransformer):
    def __init__(
        self,
        n_components,
        kernel="linear",
        lambda_=1.0,
        mu=1.0,
        eta=1.0,
        aug=False,
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

    def fit(self, x, y=None, groups=None):
        """
        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Labels, shape (nl_samples,)
        groups : array-like
            Domain/group information or covariates, shape (n_samples, n_co-variates)

        Note
        ----
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        """
        if self.aug and isinstance(groups, np.ndarray):
            x = np.concatenate((x, groups), axis=1)
        ker_x = self._get_kernel(x)
        n_samples = x.shape[0]
        unit_mat, ctr_mat = self._get_unit_ctr_mat(n_samples)
        if isinstance(groups, np.ndarray):
            ker_c = np.dot(groups, groups.T)
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
        idx_sorted = eig_values.argsort()

        self.eig_vectors = eig_vectors[:, idx_sorted][:, : self.n_components]
        self.eig_vectors = np.asarray(self.eig_vectors, dtype=np.double)
        self.eig_values = eig_values[idx_sorted][: self.n_components]

        if self.fit_inverse_transform:
            # scaled_eig_vec = self.eig_vectors / np.sqrt(np.abs(self.eig_values))
            # x_transformed = np.dot(ker_x, scaled_eig_vec)
            x_transformed = np.dot(ker_x, self.eig_vectors)
            self._fit_inverse_transform(x_transformed, x)

        self.x_fit = x
        return self

    def transform(self, x, groups=None):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)
        groups : array-like,
            Domain co-variates, shape (n_samples, n_co-variates)
        Returns
        -------
        array-like
            transformed data
        """
        check_is_fitted(self, "x_fit")
        if self.aug and isinstance(groups, np.ndarray):
            x = np.concatenate((x, groups), axis=1)
        ker_x = self._get_kernel(x, self.x_fit)
        # scaled_eig_vec = self.eig_vectors / np.sqrt(np.abs(self.eig_values))

        # return np.dot(ker_x, scaled_eig_vec)
        return np.dot(ker_x, self.eig_vectors)

    def fit_transform(self, x, y=None, co_variates=None):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)
        y : array-like
            shape (n_samples,)
        co_variates : array-like
            shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            transformed x_transformed
        """
        self.fit(x, y, co_variates)

        return self.transform(x)
