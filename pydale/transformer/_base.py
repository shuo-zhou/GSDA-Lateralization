from scipy import linalg
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels


class _BaseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, kernel='linear', n_jobs=1.0, alpha=1.0, kernel_params=None, fit_inverse_transform=False):

        self.kernel = kernel
        self.n_jobs = n_jobs
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform

    def _get_kernel(self, x, y=None):
        if self.kernel in ['linear', 'rbf', 'poly']:
            params = self.kernel_params or {}
        else:
            raise ValueError('Pre-computed kernel not supported')
        return pairwise_kernels(x, y, metric=self.kernel,
                                filter_params=True, n_jobs=self.n_jobs,
                                **params)

    def _fit_inverse_transform(self, x_transformed, x):
        if hasattr(x, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for "
                                      "sparse matrices!")

        n_samples = x_transformed.shape[0]
        ker = self._get_kernel(x_transformed)
        ker.flat[::n_samples + 1] += self.alpha
        self.dual_coef_ = linalg.solve(ker, x, sym_pos=True, overwrite_a=True)
        self.X_transformed_fit_ = x_transformed

    def inverse_transform(self, x):
        """Transform X back to original space.
        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_components)
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        if not self.fit_inverse_transform:
            raise ValueError("The fit_inverse_transform parameter was not set to True when instantiating and hence "
                             "the inverse transform is not available.")

        ker_x = self._get_kernel(x, self.X_transformed_fit_)
        n_samples = self.X_transformed_fit_.shape[0]
        ker_x.flat[::n_samples + 1] += self.alpha
        return np.dot(ker_x, self.dual_coef_)

    @staticmethod
    def _get_unit_ctr_mat(n_):
        unit_mat = np.eye(n_)
        # Construct centering matrix
        ctr_mat = unit_mat - 1. / n_ * np.ones((n_, n_))

        return unit_mat, ctr_mat
