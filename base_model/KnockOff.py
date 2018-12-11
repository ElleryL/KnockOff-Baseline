import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        '''

        Given (X,Z) -> X_tilde
        :param x_dim: int
        :param h_dim: [units,units,units,...,units]
        :param z_dim: int
        '''
        super(Generator, self).__init__()

        self.net = []
        hs = [x_dim + z_dim] + h_dim + [x_dim]
        for h0,h1 in zip(hs,hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
    def forward(self,x):
        return self.net(x)

class f_MLP(nn.Module):
    '''
    f:X -> R
    '''
    def __init__(self, x_dim, h_dim):
        '''

        Given (X) -> real scalar
        :param x_dim: int
        :param h_dim: [units,units,units,...,units]
        '''
        super(f_MLP, self).__init__()

        self.net = []
        hs = [x_dim] + h_dim + [1]
        for h0,h1 in zip(hs,hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
    def forward(self,x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim, out_dim):
        '''

        Given (X,X_tilde) -> estimated_S
        :param x_dim: int
        :param h_dim: [units,units,units,...,units]
        :param out_dim: int
        '''
        super(Discriminator, self).__init__()

        self.net = []
        hs = [x_dim * 2] + h_dim + [out_dim]
        for h0,h1 in zip(hs,hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.Sigmoid(), # map to (0,1)
            ])
        self.net = nn.Sequential(*self.net)
    def forward(self,x):
        return self.net(x)

    def loss(self,S,estimated_S):
        loss = S * torch.log(estimated_S) + (1 - S) * torch.log(1 - estimated_S)
        return loss.sum(-1).mean(-1).sum()

class MINE(nn.Module):

    def __init__(self, h_dim):
        '''
        Given (X,X_tilde) -> Mutual Information

        :param h_dim:

        '''
        super(MINE, self).__init__()

        self.net_x = []
        hs = [1] + h_dim + [1]
        for h0,h1 in zip(hs,hs[1:]):
            self.net_x.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net_x.pop()  # pop the last ReLU for the output layer
        self.net_x = nn.Sequential(*self.net_x)

        self.net_x_tilde = []
        hs = [1] + h_dim + [1]
        for h0,h1 in zip(hs,hs[1:]):
            self.net_x_tilde.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net_x_tilde.pop()  # pop the last ReLU for the output layer
        self.net_x_tilde = nn.Sequential(*self.net_x_tilde)

        self.out = nn.Linear(hs[-1], 1)

    def forward(self,X):
        d = int(X.shape[-1] / 2)
        x,x_tilde = X[:,:d], X[:,d:]
        h1 = F.relu(self.net_x(x) + self.net_x_tilde(x_tilde))
        h2 = self.out(h1)
        return h2




def random_swap(X,X_tilde,batch_size):
    '''
    randomly generate S {0,1}^D
    '''
    N = X.shape[0]
    D = X.shape[-1]
    S = torch.Tensor(batch_size,N, D).random_(2)
    x_swaped = S * X_tilde + (1 - S) * X
    x_tilde_swaped = S * X + (1 - S) * X_tilde
    return torch.cat((x_swaped, x_tilde_swaped), -1),S
