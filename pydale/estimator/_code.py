import torch
import torchvision
from time import time
import torch.nn as nn
from torch.nn import functional as F


def hsic(x, y):
    kx = torch.mm(x, x.T)
    ky = torch.mm(y, y.T)
    
    n = x.shape[0]
    ctr_mat = torch.eye(n) - torch.ones((n, n)) / n
    
    return torch.trace(torch.mm(torch.mm(torch.mm(kx, ctr_mat), ky), ctr_mat)) / (n ** 2)


class CoDeLR(nn.Module):
    def __init__(self, lr=0.001, l1_hparam=0.0, l2_hparam=1.0, lambda_=1.0, n_epochs=500):
        super().__init__()
        self.lr = lr
        self.l1_hparam = l1_hparam
        self.l2_hparam = l2_hparam
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.model: nn.modules.Linear
        self.optimizer: torch.optim.Adam
        self.fit_time = 0
        self.losses = {'ovr': [], 'pred': [], 'code': [], 'time': []}
        # self.linear = nn.Linear(n_features, n_classes)
        
    def fit(self, x, y, c, train_idx=None):
        n_features = x.shape[1]
        n_classes = torch.unique(y).shape[0]
        # self.model = nn.Linear(n_features, n_classes)
        self.model = nn.Linear(n_features, 1)
        n_train = y.shape[0]
        if train_idx is None:
            train_idx = torch.arange(n_train)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_hparam)

        start_time = time()
        for epoch in range(self.n_epochs):
            out = self.model(x)
            pred_loss = self._compute_pred_loss(out[train_idx], y.view(-1))
            code_loss = self._compute_code_loss(out, c)
            loss = pred_loss + self.lambda_ * code_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            if (epoch+1) % 10 == 0:
                time_used = time() - start_time
                self.losses['ovr'].append(loss.item())
                self.losses['pred'].append(pred_loss.item())
                self.losses['code'].append(code_loss.item())
                self.losses['time'].append(time_used)
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
