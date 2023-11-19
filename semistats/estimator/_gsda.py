from time import time

import numpy as np
import torch
from numpy.linalg import multi_dot
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.nn import functional as F


def hsic(x, y):
    kx = torch.mm(x, x.T)
    ky = torch.mm(y, y.T)

    n = x.shape[0]
    ctr_mat = torch.eye(n) - torch.ones((n, n)) / n

    return torch.trace(torch.mm(torch.mm(torch.mm(kx, ctr_mat), ky), ctr_mat)) / (
        n**2
    )


def simple_hsic(w, x, groups):
    n = x.shape[0]
    kernel_c = np.dot(groups, groups.T)
    ctr_mat = np.diag(np.ones(n)) - 1 / n

    return multi_dot((x.T, ctr_mat, kernel_c, ctr_mat, x, w))


def gsda_nll(w, x, y, groups, tgt_idx=None, alpha=1.0, lambda_=1.0):
    n_tgt = y.shape[0]
    if isinstance(tgt_idx, np.ndarray):
        x_tgt = x[tgt_idx]
    else:
        x_tgt = x[:n_tgt]
    # y_hat = expit(x_tgt @ w)
    # errors = y_hat - y
    _simple_hsic = simple_hsic(w, x, groups)
    nll = (
        (
            np.dot(y, np.log(expit(np.dot(x_tgt, w))))
            + np.dot((1 - y), np.log(1 - expit(np.dot(x_tgt, w))))
        )
        / n_tgt
        + alpha * np.dot(w, w)
        + lambda_ * expit(np.dot(w, _simple_hsic))
    )

    return nll


def _compute_pred_loss(y, y_hat):
    pred_log_loss = (
        -1
        * (np.dot(y, np.log(y_hat + 1e-6)) + np.dot((1 - y), np.log(1 - y_hat + 1e-6)))
        / y.shape[0]
    )
    return pred_log_loss


class GSLR(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Classifier
    Parameters
    ----------
    learning_rate : int or float, default=0.1
        The tuning parameter for the optimization algorithm (here, Gradient Descent)
        that determines the step size at each iteration while moving toward a minimum
        of the cost function.
    max_iter : int, default=100
        Maximum number of iterations taken for the optimization algorithm to converge

    regularization : None or 'l2', default='l2'.
        Option to perform L2 regularization.
    alpha : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.
    tolerance : float, optional, default=1e-4
        Value indicating the weight change between epochs in which
        gradient descent should be terminated.
    """

    def __init__(
        self,
        learning_rate=0.1,
        max_iter=100,
        regularization="l2",
        alpha=1.0,
        tolerance=1e-4,
        lambda_=1.0,
        solver="gd",
        memory_size=10,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.alpha = alpha
        self.tolerance = tolerance
        self.theta = None
        self.lambda_ = lambda_
        self.losses = {"ovr": [], "pred": [], "hsic": [], "time": []}
        self.solver = solver
        self.memory_size = memory_size

    def fit(self, x, y, groups, target_idx=None):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        groups: array-like
        target_idx: array-like
        Returns
        -------
        self : object
        """
        x = np.asarray(x)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if groups.ndim == 1:
            groups = groups.reshape((-1, 1))
        self.theta = np.random.random((x.shape[1] + 1))
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        # n_tgt = y.shape[0]
        # n_sample = x.shape[0]
        # if target_idx is None:
        #     x_tgt = x[:n_tgt]
        # else:
        #     x_tgt = x[target_idx]

        start_time = time()
        if self.solver == "lbfgs":
            self.lbfgs(x, y, groups, target_idx)
            # impltmentations of L-BFGS
            # res = minimize(
            #     gsda_nll,
            #     self.theta,
            #     args=(x, y, groups, target_idx, self.alpha, self.lambda_),
            #     method="L-BFGS-B",
            #     jac=False,
            #     options={"maxiter": self.max_iter, "disp": True},
            # )
            # self.theta = res.x
            #
            # time_used = time() - start_time
            # _simple_hsic = simple_hsic(self.theta, x, groups)
            # hsic_proba = self._compute_hsic_proba(_simple_hsic, n_sample)
            # hsic_log_loss = -1 * np.log(hsic_proba)
            # y_hat = expit(x_tgt @ self.theta)
            # pred_log_loss = _compute_pred_loss(y, y_hat)
            # self.losses["ovr"].append(pred_log_loss + hsic_log_loss)
            # self.losses["pred"].append(pred_log_loss)
            # self.losses["hsic"].append(hsic_log_loss)
            # self.losses["time"].append(time_used)
        else:
            for _ in range(self.max_iter):
                # y_hat = self.__sigmoid(x_tgt @ self.theta)
                # errors = y_hat - y
                # n_feature = x.shape[1]
                # _simple_hsic = simple_hsic(self.theta, x, groups)
                # hsic_proba = self._compute_hsic_proba(_simple_hsic, n_sample)
                # hsic_log_loss = -1 * np.log(hsic_proba)
                # grad_hsic = (hsic_proba - 1) * _simple_hsic / np.square(n_sample - 1)
                #
                # if self.regularization is not None:
                #     delta_grad = (
                #         (x_tgt.T @ errors) / n_tgt
                #         + self.theta / self.alpha
                #         + self.lambda_ * grad_hsic
                #     )
                # else:
                #     delta_grad = x_tgt.T @ errors
                # pred_log_loss = _compute_pred_loss(y, y_hat)
                delta_grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(
                    x, y, groups, target_idx
                )
                if _ % 10 == 0:
                    time_used = time() - start_time
                    self.losses["ovr"].append(pred_log_loss + hsic_log_loss)
                    self.losses["pred"].append(pred_log_loss)
                    self.losses["hsic"].append(hsic_log_loss)
                    self.losses["time"].append(time_used)
                    # if np.abs(self.losses["ovr"][-1] - self.losses["ovr"][-2]) < self.tolerance:
                    #     break
                    if len(self.losses["ovr"]) > 10:
                        if (
                            self.losses["ovr"][-1] > self.losses["ovr"][-2]
                            or self.losses["ovr"][-2] - self.losses["ovr"][-1]
                            < self.tolerance
                        ):
                            break

                if _ % 50 == 0 and self.learning_rate > 0.001:
                    self.learning_rate *= 0.8

                # self.theta -= self.learning_rate * delta_grad
                if not np.all(abs(delta_grad) <= self.tolerance):
                    self.theta -= self.learning_rate * delta_grad
                else:
                    break

        return self

    def predict_proba(self, x):
        """
        Probability estimates for samples in X.
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        probs : array-like of shape (n_samples,)
            Returns the probability of each sample.
        """
        return self.__sigmoid((x @ self.theta[1:]) + self.theta[0])

    def predict(self, x):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        x : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        labels : array, shape [n_samples]
            Predicted class label per sample.
        """
        y_proba = self.predict_proba(x)
        y_pred = np.zeros(y_proba.shape)
        y_pred[np.where(y_proba > 0.5)] = 1

        return y_pred

    @staticmethod
    def __sigmoid(z):
        """
        The sigmoid function.
        Parameters
        ------------
        z : float
            linear combinations of weights and sample features
            z = w_0 + w_1*x_1 + ... + w_n*x_n
        Returns
        ---------
        Value of logistic function at z
        """
        return 1 / (1 + np.exp(-z))

    def get_params(self, **kwargs):
        """
        Get method for models coefficients and intercept.
        Returns
        -------
        params : dict
        """
        try:
            params = dict()
            params["intercept"] = self.theta[0]
            params["coef"] = self.theta[1:]
            return params
        except self.theta is None:
            raise Exception("Fit the model first!")

    def _compute_hsic_proba(self, simple_hsic_, n_sample):
        hsic_score = multi_dot((self.theta, simple_hsic_)) / np.square(n_sample - 1)
        return self.__sigmoid(hsic_score)

    def lbfgs(self, x, y, groups, target_idx=None):
        old_dirs = []  # Δx
        old_stps = []  # Δgrad
        H0 = np.eye(self.theta.shape[0])  # Initial approximation of the inverse Hessian
        grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(
            x, y, groups, target_idx
        )

        for _ in range(self.max_iter):
            q = grad
            alphas = []

            theta_old = self.theta.copy()

            # Compute the search direction
            for i in reversed(range(len(old_dirs))):
                alpha = old_dirs[i].dot(q) / old_stps[i].dot(old_dirs[i])
                alphas.append(alpha)
                q = q - alpha * old_stps[i]

            r = H0.dot(q)

            for i in range(len(old_dirs)):
                beta = old_stps[i].dot(r) / old_stps[i].dot(old_dirs[i])
                r = r + old_dirs[i] * (alphas[len(alphas) - 1 - i] - beta)

            # Line search and update x
            # Implement a line search algorithm to find an appropriate step size
            step_size = 1  # Placeholder
            self.theta = theta_old + step_size * r

            # Update memory
            if len(old_dirs) == self.memory_size:
                old_dirs.pop(0)
                old_stps.pop(0)

            old_dirs.append(self.theta - theta_old)
            grad_new, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(
                x, y, groups, target_idx
            )
            old_stps.append(grad_new - grad)
            grad = np.copy(grad_new)

            # Check for convergence (implement convergence criteria)

    def compute_gsda_gradient(self, x, y, groups, target_idx=None):
        n_sample = x.shape[0]
        n_tgt = y.shape[0]
        if target_idx is None:
            x_tgt = x[:n_tgt]
        else:
            x_tgt = x[target_idx]

        y_hat = self.__sigmoid(x_tgt @ self.theta)
        errors = y_hat - y
        # n_feature = x.shape[1]
        _simple_hsic = simple_hsic(self.theta, x, groups)
        hsic_proba = self._compute_hsic_proba(_simple_hsic, n_sample)
        grad_hsic = (hsic_proba - 1) * _simple_hsic / np.square(n_sample - 1)

        if self.regularization is not None:
            delta_grad = (
                (x_tgt.T @ errors) / n_tgt
                + self.theta / self.alpha
                + self.lambda_ * grad_hsic
            )
        else:
            delta_grad = x_tgt.T @ errors

        hsic_log_loss = -1 * np.log(hsic_proba)
        pred_log_loss = _compute_pred_loss(y, y_hat)

        return delta_grad, pred_log_loss, hsic_log_loss

    # @staticmethod
    # def _hsic_score(w, x, groups):
    #     n = x.shape[0]
    #     kernel_c = np.dot(groups, groups.T)
    #     ctr_mat = np.diag(np.ones(n)) - 1 / n
    #
    #     return multi_dot((w, x.T, ctr_mat, kernel_c, ctr_mat, x, w.T)) / np.square(  # type: ignore
    #         n - 1
    #     )


class GSLR_Torch(nn.Module):
    def __init__(
        self, lr=0.001, l1_hparam=0.0, l2_hparam=1.0, lambda_=1.0, n_epochs=500
    ):
        super().__init__()
        self.lr = lr
        self.l1_hparam = l1_hparam
        self.l2_hparam = l2_hparam
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.model = None
        self.optimizer = None
        self.fit_time = 0
        self.losses = {"ovr": [], "pred": [], "code": [], "time": []}
        # self.linear = nn.Linear(n_features, n_classes)

    def fit(self, x, y, c, target_idx=None):
        n_features = x.shape[1]
        # n_classes = torch.unique(y).shape[0]
        # self.model = nn.Linear(n_features, n_classes)
        self.model = nn.Linear(n_features, 1)
        n_train = y.shape[0]
        if target_idx is None:
            target_idx = torch.arange(n_train)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_hparam
        )
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        start_time = time()
        for epoch in range(self.n_epochs):
            out = self.model(x)
            pred_loss = self._compute_pred_loss(out[target_idx], y.view(-1))
            # l2_norm = torch.linalg.norm()
            code_loss = self._compute_code_loss(out, c)
            loss = pred_loss + self.lambda_ * code_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                time_used = time() - start_time
                self.losses["ovr"].append(loss.item())
                self.losses["pred"].append(pred_loss.item())
                self.losses["code"].append(code_loss.item())
                self.losses["time"].append(time_used)
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.n_epochs, loss.item()))
                # print('Pred Loss: {:.4f}'.format(pred_loss.item()))
                # print('CoDe Loss: {:.4f}'.format(code_loss.item()))

        # print('Time used: {:.4f}'.format(time_used))

    @staticmethod
    def _compute_pred_loss(input_, y):
        # return F.cross_entropy(F.sigmoid(input_), y.view(-1))
        return F.binary_cross_entropy(torch.sigmoid(input_), y.view((-1, 1)).float())

    @staticmethod
    def _compute_code_loss(input_, c):
        return 1 - torch.sigmoid(hsic(input_, c.float()))

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        output = self.forward(x)
        # _, y_pred = torch.max(output, 1)
        y_prob = torch.sigmoid(output)
        y_pred = torch.zeros(output.shape)
        y_pred[torch.where(y_prob > 0.5)] = 1

        return y_pred
