# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from scipy import linalg
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from ..utils import lap_norm, mmd_coef
from ._base import _BaseTransformer


class TCA(_BaseTransformer):
    def __init__(
        self,
        n_components,
        kernel="linear",
        lambda_=1.0,
        mu=1.0,
        gamma_=0.5,
        n_jobs=1.0,
        k=3,
        alpha=1.0,
        kernel_params=None,
        fit_inverse_transform=False,
        **kwargs
    ):
        """Transfer Component Analysis: TCA

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= (N, d))
        kernel: str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        lambda_ : float
            regulisation param
        mu : float
            KNN graph param
        k : int
            number of nearest neighbour for KNN graph
        gamma : float
            label dependence param

        References
        ----------
        S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
        Transfer Component Analysis," in IEEE Transactions on Neural Networks,
        vol. 22, no. 2, pp. 199-210, Feb. 2011.
        """
        super().__init__(kernel, n_jobs, alpha, kernel_params, fit_inverse_transform)
        self.n_components = n_components
        self.kwargs = kwargs
        self.lambda_ = lambda_
        self.mu = mu
        self.gamma_ = gamma_
        self.k = k
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)
        self.xs = None
        self.xt = None

    def fit(self, xs, ys=None, xt=None, yt=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features)
        xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None
        """
        if isinstance(xt, np.ndarray):
            x = np.vstack((xs, xt))
            ns = xs.shape[0]
            nt = xt.shape[0]
            L = mmd_coef(ns, nt, kind="marginal", mu=0)
            L[np.isnan(L)] = 0
        else:
            x = xs
            L = np.zeros((x.shape[0], x.shape[0]))

        ker_x = self._get_kernel(x)
        n_samples = x.shape[0]
        unit_mat, ctr_mat = self._get_unit_ctr_mat(n_samples)

        obj_min = self.lambda_ * unit_mat
        obj_max = multi_dot([ker_x, ctr_mat, ker_x.T])
        if ys is not None:
            # semisupervised TCA (SSTCA)
            ys_mat = self._lb.fit_transform(ys)
            n_class = ys_mat.shape[1]
            y = np.zeros((n_samples, n_class))
            y[: ys_mat.shape[0], :] = ys_mat[:]
            if yt is not None:
                yt_mat = self._lb.transform(yt)
                y[ys_mat.shape[0] : yt_mat.shape[0], :] = yt_mat[:]
            ker_y = self.gamma_ * np.dot(y, y.T) + (1 - self.gamma_) * unit_mat
            lap_mat = lap_norm(x, n_neighbour=self.k, mode="connectivity")
            obj_min += multi_dot([ker_x, (L + self.mu * lap_mat), ker_x.T])
            obj_max += multi_dot([ker_x, ctr_mat, ker_y, ctr_mat, ker_x.T])
        # obj_min = np.trace(np.dot(ker_x,L))
        else:
            obj_min += multi_dot([ker_x, ctr_mat, L, ker_x.T])

        objective = np.dot(linalg.inv(obj_min), obj_max)
        eig_values, eig_vectors = linalg.eigh(objective)
        idx_sorted = (-1 * eig_values).argsort()

        self.eig_vectors = eig_vectors[:, idx_sorted][:, : self.n_components]
        self.eig_vectors = np.asarray(self.eig_vectors, dtype=np.double)
        self.eig_values = eig_values[idx_sorted][: self.n_components]

        if self.fit_inverse_transform:
            scaled_eig_vec = self.eig_vectors / np.sqrt(self.eig_values)
            x_transformed = np.dot(ker_x, scaled_eig_vec)
            self._fit_inverse_transform(x_transformed, x)

        self.xs = xs
        self.xt = xt

        return self

    def transform(self, x):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            transformed data
        """
        check_is_fitted(self, "xs")
        check_is_fitted(self, "xt")
        x_fit = np.vstack((self.xs, self.xt))
        ker_x = self._get_kernel(x, x_fit)
        scaled_eig_vec = self.eig_vectors / np.sqrt(self.eig_values)

        return np.dot(ker_x, scaled_eig_vec)

    def fit_transform(self, xs, ys=None, xt=None, yt=None):
        """
        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features).
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None.
        xt : array-like
            Target domain data, shape (nt_samples, n_features), by default None.
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None.

        Returns
        -------
        array-like
            transformed xs_transformed, xt_transformed
        """
        self.fit(xs, ys, xt, yt)

        return self.transform(xs), self.transform(xt)
