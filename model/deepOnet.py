import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import tqdm
import torch.nn.functional as F
from scipy.interpolate import griddata
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.hidden_neurons = layers
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.0)
            self.hidden_layers.append(
                layer
            )

    def forward(self, x):
        for hidden_layer in self.hidden_layers[:-1]:
            x = self.activation(hidden_layer(x))
        return self.hidden_layers[-1](x)



class DeepONet2D(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet2D, self).__init__()
        self.branch_net = MLP(branch_net)
        self.trunk_net = MLP(trunk_net)
        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load

    def forward(self, s, y):
        B = self.branch_net(s)
        # y = torch.stack([x, t], dim=1)
        T = self.trunk_net(y)
        return torch.sum(B * T, dim=-1, keepdim=True)


class DeepONet3D(nn.Module):
    def __init__(self, ):
        super(DeepONet3D, self).__init__()
        n_c = 5
        num_point_layer = 3
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)  # (B, 32, N)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)  # (B, 64, N)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)  # (B, 128, N)
        self.bn3 = nn.BatchNorm1d(128)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 新增 trunk_net 的 Linear 层
        self.trunk_linear1 = nn.Linear(3, 32)
        self.trunk_linear2 = nn.Linear(32, 64)
        self.trunk_linear3 = nn.Linear(64, 128)
        self.trunk_final1 = nn.Linear(129, 128)
        self.trunk_final2 = nn.Linear(128, 256)
        self.trunk_final3 = nn.Linear(256, 128)

        self.branch_linear1 = nn.Linear(n_c, 128)
        self.branch_linear2 = nn.Linear(128, 256)
        self.branch_linear3 = nn.Linear(256, 128)

        # self.branch_linear4 = nn.Linear(128 * N, 128)
        self.branch_linear5 = nn.Linear(128, 256)
        self.branch_linear6 = nn.Linear(256, 128)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save

    def trunk_net(self, X):
        B, N, _ = X.shape
        # SIREN layer
        x1, x2 = X[:, :, :3], X[:, :, 3:4]  # (B, N, 3), # (B, N, 1)
        x1 = x1.reshape(B * N, -1)  # (B*N, 3)
        x1 = self.trunk_linear1(x1)  # Project from 3 to 32 features
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear2(x1)  # (B*N, 64)
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear3(x1)  # (B*N, 128)
        x1 = self.sigma2(x1)  # sine activation
        x1 = x1.reshape(B, N, 128)  # (B, N, 128)

        # Concatenate with x2
        x_cat = torch.cat([x1, x2], dim=-1)  # (B, N, 129)

        # Process concatenated features
        x_cat = x_cat.reshape(B * N, 129)  # (B*N, 129)
        x3 = self.trunk_final1(x_cat)  # (B*N, 128)
        x4 = self.sigma3(x3)
        x4 = self.trunk_final2(x4)  # (B*N, 256)
        x4 = self.sigma3(x4)
        x4 = self.trunk_final3(x4)  # (B*N, 512)
        x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 128)  # (B, N, 512)
        T = x4.reshape(B, N, 128, 1)
        # B = nn.Parameter(torch.randn(128)).unsqueeze(0).expand(B, -1)
        # out = (T * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # (B, N, 4)
        # out = torch.tanh(out)
        return T, x3

    def branch_net(self, C, X, x3):
        x1, x2 = C, X
        n_b, n_c = C.shape
        N = X.shape[1]
        x1 = self.branch_linear1(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear2(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear3(x1)
        x1 = self.sigma(x1)

        # x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # # 第一层: Conv1d + BN + SiLU
        # x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # # 第二层: Conv1d + BN + SiLU
        # x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # # 第三层: Conv1d + BN + SiLU
        # x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # # 全局最大池化 (沿最后一个维度 N)
        # x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        # x = x1 + x2  # (B, 128)
        x = x1  # (B, 128)
        # Element-wise multiplication
        # x = x.unsqueeze(1).expand(-1, N, -1).reshape(n_b * N, 128)
        # x = x3 * x  # (B*N, 128)
        # x = x.reshape(n_b, -1) # (B*N, 128)
        # x = self.branch_linear4(x)  # (128*N, 128)
        x = self.sigma(x)
        # To get (B, 128), we can sum over the N dimension (assuming permutation invariance)
        x = self.branch_linear5(x)
        x = self.sigma(x)
        B = self.branch_linear6(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        out = (T * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.tanh(out)
        return out



class DeepONet3D_multi_u(nn.Module):
    def __init__(self, ):
        super(DeepONet3D_multi_u, self).__init__()
        n_c = 5
        num_point_layer = 3
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)  # (B, 32, N)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)  # (B, 64, N)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)  # (B, 128, N)
        self.bn3 = nn.BatchNorm1d(128)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 新增 trunk_net 的 Linear 层
        self.trunk_linear1 = nn.Linear(3, 32)
        self.trunk_linear2 = nn.Linear(32, 64)
        self.trunk_linear3 = nn.Linear(64, 128)
        self.trunk_final1 = nn.Linear(129, 128)
        self.trunk_final2 = nn.Linear(128, 256)
        self.trunk_final3 = nn.Linear(256, 512)

        self.branch_linear1 = nn.Linear(n_c, 128)
        self.branch_linear2 = nn.Linear(128, 256)
        self.branch_linear3 = nn.Linear(256, 128)

        # self.branch_linear4 = nn.Linear(128 * N, 128)
        self.branch_linear5 = nn.Linear(128, 256)
        self.branch_linear6 = nn.Linear(256, 128)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load
    def trunk_net(self, X):
        B, N, _ = X.shape
        # SIREN layer
        x1, x2 = X[:, :, :3], X[:, :, 3:4]  # (B, N, 3), # (B, N, 1)
        x1 = x1.reshape(B * N, -1)  # (B*N, 3)
        x1 = self.trunk_linear1(x1)  # Project from 3 to 32 features
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear2(x1)  # (B*N, 64)
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear3(x1)  # (B*N, 128)
        x1 = self.sigma2(x1)  # sine activation
        x1 = x1.reshape(B, N, 128)  # (B, N, 128)

        # Concatenate with x2
        x_cat = torch.cat([x1, x2], dim=-1)  # (B, N, 129)

        # Process concatenated features
        x_cat = x_cat.reshape(B * N, 129)  # (B*N, 129)
        x3 = self.trunk_final1(x_cat)  # (B*N, 128)
        x4 = self.sigma3(x3)
        x4 = self.trunk_final2(x4)  # (B*N, 256)
        x4 = self.sigma3(x4)
        x4 = self.trunk_final3(x4)  # (B*N, 512)
        x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 512)  # (B, N, 512)
        T = x4.reshape(B, N, 128, 4)
        return T, x3

    def branch_net(self, C, X, x3):
        x1, x2 = C, X
        n_b, n_c = C.shape
        N = X.shape[1]
        x1 = self.branch_linear1(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear2(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear3(x1)
        x1 = self.sigma(x1)
        x = x1
        x = self.sigma(x)
        x = self.branch_linear5(x)
        x = self.sigma(x)
        B = self.branch_linear6(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        out = (T * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.tanh(out)
        return out


class DeepONet3D_multi_u_v2(nn.Module):
    def __init__(self, ):
        super(DeepONet3D_multi_u_v2, self).__init__()
        n_c = 5
        num_point_layer = 3
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)  # (B, 32, N)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)  # (B, 64, N)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)  # (B, 128, N)
        self.bn3 = nn.BatchNorm1d(128)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 新增 trunk_net 的 Linear 层
        self.trunk_linear1 = nn.Linear(3, 32)
        self.trunk_linear2 = nn.Linear(32, 64)
        self.trunk_linear3 = nn.Linear(64, 128)
        self.trunk_final1 = nn.Linear(129, 128)
        self.trunk_final2 = nn.Linear(128, 256)
        self.trunk_final3 = nn.Linear(256, 512)
        self.trunk_final4 = nn.Linear(512, 1024)
        self.branch_linear1 = nn.Linear(n_c, 128)
        self.branch_linear2 = nn.Linear(128, 256)
        self.branch_linear3 = nn.Linear(256, 128)

        # self.branch_linear4 = nn.Linear(128 * N, 128)
        self.branch_linear5 = nn.Linear(128, 256)
        self.branch_linear6 = nn.Linear(256, 128)
        self.branch_linear7 = nn.Linear(128, 32)

        self.out_linear1 = nn.Linear(32, 4)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save

    def trunk_net(self, X):
        B, N, _ = X.shape
        # SIREN layer
        x1, x2 = X[:, :, :3], X[:, :, 3:4]  # (B, N, 3), # (B, N, 1)
        x1 = x1.reshape(B * N, -1)  # (B*N, 3)
        x1 = self.trunk_linear1(x1)  # Project from 3 to 32 features
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear2(x1)  # (B*N, 64)
        x1 = self.sigma2(x1)  # sine activation
        x1 = self.trunk_linear3(x1)  # (B*N, 128)
        x1 = self.sigma2(x1)  # sine activation
        x1 = x1.reshape(B, N, 128)  # (B, N, 128)

        # Concatenate with x2
        x_cat = torch.cat([x1, x2], dim=-1)  # (B, N, 129)

        # Process concatenated features
        x_cat = x_cat.reshape(B * N, 129)  # (B*N, 129)
        x3 = self.trunk_final1(x_cat)  # (B*N, 128)
        x4 = self.sigma3(x3)
        x4 = self.trunk_final2(x4)  # (B*N, 256)
        x4 = self.sigma3(x4)
        x4 = self.trunk_final3(x4)  # (B*N, 512)
        x4 = self.sigma3(x4)
        x4 = self.trunk_final4(x4)
        x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 1024)  # (B, N, 512)
        T = x4.reshape(B, N, 32, 32)
        return T, x3

    def branch_net(self, C, X, x3):
        x1, x2 = C, X
        n_b, n_c = C.shape
        N = X.shape[1]
        x1 = self.branch_linear1(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear2(x1)
        x1 = self.sigma(x1)
        x1 = self.branch_linear3(x1)
        x1 = self.sigma(x1)
        x = x1
        x = self.sigma(x)
        x = self.branch_linear5(x)
        x = self.sigma(x)
        x = self.branch_linear6(x)
        B = self.branch_linear7(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        out = (T * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.tanh(out)
        out = self.out_linear1(out)
        return out