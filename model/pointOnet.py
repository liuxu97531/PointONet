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

class Pointlayer(nn.Module):
    def __init__(self, neuron, global_feat=True, channel=3):
        super(Pointlayer, self).__init__()
        self.neuron = neuron
        self.conv1 = torch.nn.Conv1d(channel, self.neuron, 1)
        self.conv2 = torch.nn.Conv1d(self.neuron, self.neuron, 1)
        self.conv3 = torch.nn.Conv1d(self.neuron, self.neuron, 1)

        self.mlp = MLP([self.neuron, self.neuron, self.neuron])

        self.bn1 = nn.ReLU()
        self.bn2 = nn.ReLU()
        self.bn3 = nn.ReLU()
        self.global_feat = global_feat

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.neuron)
        return self.mlp(x)



# class PointONet2D(nn.Module):
#     def __init__(self, loc_branch_neuron, branch_net_channel, trunk_net):
#         super(PointONet2D, self).__init__()
#         self.branch_net = Pointlayer(neuron = loc_branch_neuron, channel=branch_net_channel)
#         self.trunk_net = MLP(trunk_net)
#         def save(filepath):
#             torch.save(self.state_dict(), filepath)
#         self.save = save
#         def load(filename):
#             self.load_state_dict(torch.load(filename))
#         self.load = load
#
#     def forward(self, s, y):
#         B = self.branch_net(s)
#         # y = torch.stack([x, t], dim=1)
#         T = self.trunk_net(y)
#         return torch.sum(B * T, dim=-1, keepdim=True)


class PointONet2D(nn.Module):
    def __init__(self, loc_branch_neuron, branch_net_channel, trunk_net):
        super(PointONet2D, self).__init__()
        self.branch_net = Pointlayer(neuron = loc_branch_neuron, channel=branch_net_channel)
        self.trunk_net = MLP(trunk_net)
        self.linear = MLP([1, loc_branch_neuron, 1])
        self.activation = torch.tanh
        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load

    def forward(self, s, y):
        B = self.branch_net(s)
        T = self.trunk_net(y)
        out = torch.sum(B * T, dim=-1, keepdim=True)
        out = self.linear(self.activation(out))
        return out


class PointONet2D_separate(nn.Module):
    def __init__(self, value_branch_net, loc_branch_neuron, loc_branch_net_channel, trunk_net):
        super(PointONet2D_separate, self).__init__()
        self.branch_net = Pointlayer(neuron = loc_branch_neuron, channel=loc_branch_net_channel)
        self.branch_net2 = MLP(value_branch_net)
        self.trunk_net = MLP(trunk_net)
        self.branch_net3 = MLP([100, 100])
        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load

    def forward(self, s_x, y):
        x, s = s_x[:, :1, :], s_x[:, 1, :]
        B = self.branch_net(x)
        B2 = self.branch_net2(s)
        B = self.branch_net3(B + B2)

        # y = torch.stack([x, t], dim=1)
        T = self.trunk_net(y)
        return torch.sum(B * T, dim=-1, keepdim=True)


class PointONet3D(nn.Module):
    def __init__(self, ):
        super(PointONet3D, self).__init__()
        n_c = 5
        num_point_layer = 3
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)  # (B, 32, N)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)  # (B, 64, N)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)  # (B, 128, N)
        self.bn3 = nn.BatchNorm1d(128)

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
        x4 = x4.reshape(B, N, 128)  # (B, N, 512)
        T = x4.reshape(B, N, 128, 1)
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
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

# 主干网络与分支网络点积后，接一层MLP[32, 4]
class PointONet3D_multi_u(nn.Module):
    def __init__(self, ):
        super(PointONet3D_multi_u, self).__init__()
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
        # Element-wise multiplication
        # x = x.unsqueeze(1).expand(-1, N, -1).reshape(n_b * N, 128)
        # x = x3 * x  # (B*N, 128)
        # x = x.reshape(n_b, -1) # (B*N, 128)
        # x = self.branch_linear4(x)  # (128*N, 128)
        x = self.sigma(x)
        # To get (B, 128), we can sum over the N dimension (assuming permutation invariance)
        x = self.branch_linear5(x)
        x = self.sigma(x)
        x = self.branch_linear6(x)
        x = self.sigma(x)
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



class PointONet3D_multi_u_v2(nn.Module):
    def __init__(self, ):
        super(PointONet3D_multi_u_v2, self).__init__()
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
        self.trunk_final3 = nn.Linear(256, 384)

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []

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
        x4 = x4.reshape(B, N, 384)  # (B, N, 512)
        T = x4.reshape(B, N, 128, 3)
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
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


# 同一个主干网络输出分别与分支网络四个输出点积得到四个输出
class PointONet3D_multi_u_v5(nn.Module):
    def __init__(self, ):
        super(PointONet3D_multi_u_v5, self).__init__()
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
        self.trunk_final4 = nn.Linear(512, 128)
        self.trunk_final5 = nn.Linear(128, 32)

        self.branch_linear1 = nn.Linear(n_c, 128)
        self.branch_linear2 = nn.Linear(128, 256)
        self.branch_linear3 = nn.Linear(256, 128)

        # self.branch_linear4 = nn.Linear(128 * N, 128)
        self.branch_linear5 = nn.Linear(128, 256)
        self.branch_linear6 = nn.Linear(256, 128)
        # self.branch_linear7 = nn.Linear(128, 32)

        self.out_linear1 = nn.Linear(32, 4)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []

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
        x4 = self.trunk_final4(x4)
        x4 = self.sigma3(x4)
        x4 = self.trunk_final5(x4)
        x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 32)  # (B, N, 512)
        T = x4.reshape(B, N, 32, 1)
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
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
        # x = self.sigma(x)
        # B = self.branch_linear7(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        B1, B2, B3, B4 = B[:, :32], B[:, 32:64], B[:, 64:96], B[:, 96:]
        out1 = (T * B1.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out2 = (T * B2.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out3 = (T * B3.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out4 = (T * B4.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.concatenate([out1,out2,out3,out4],dim=2)
        out = torch.tanh(out)
        # out = self.out_linear1(out)
        return out

# 主干网络输出分别与同一个分支网络点积得到四个输出
class PointONet3D_multi_u_v6(nn.Module):
    def __init__(self, ):
        super(PointONet3D_multi_u_v6, self).__init__()
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
        self.trunk_final4 = nn.Linear(512, 256)
        # self.trunk_final5 = nn.Linear(128, 32)

        self.branch_linear1 = nn.Linear(n_c, 128)
        self.branch_linear2 = nn.Linear(128, 256)
        self.branch_linear3 = nn.Linear(256, 128)

        # self.branch_linear4 = nn.Linear(128 * N, 128)
        self.branch_linear5 = nn.Linear(128, 256)
        self.branch_linear6 = nn.Linear(256, 64)
        # self.branch_linear7 = nn.Linear(128, 32)

        self.out_linear1 = nn.Linear(32, 4)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

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
        # x4 = self.trunk_final5(x4)
        # x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 256,1)  # (B, N, 512)
        T = x4
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
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
        # x = self.sigma(x)
        # B = self.branch_linear7(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        # B1, B2, B3, B4 = B[:, :32], B[:, 32:64], B[:, 64:96], B[:, 96:]
        T1, T2, T3, T4 = T[:, :, :64, :], T[:, :,64:128, :], T[:, :,128:192, :], T[:, :, 192:, :]

        out1 = (T1 * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out2 = (T2 * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out3 = (T3 * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out4 = (T4 * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.concatenate([out1,out2,out3,out4],dim=2)
        out = torch.tanh(out)
        # out = self.out_linear1(out)
        return out

# 点积后接两层MLP[32,16,4]
class PointONet3D_multi_u_v7(nn.Module):
    def __init__(self, ):
        super(PointONet3D_multi_u_v7, self).__init__()
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

        self.out_linear1 = nn.Linear(32, 16)
        self.out_linear2 = nn.Linear(16, 4)
        self.sigma = torch.tanh
        # self.sigma2 = torch.sin
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []

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
        x4 = self.trunk_final4(x4)
        x4 = self.sigma3(x4)
        x4 = x4.reshape(B, N, 1024)  # (B, N, 512)
        T = x4.reshape(B, N, 32, 32)
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

        x2 = x2.permute(0, 2, 1)  # (B, 3, N)
        # 第一层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn1(self.conv1(x2)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn2(self.conv2(x2)))  # (B, 64, N)
        # 第三层: Conv1d + BN + SiLU
        x2 = self.sigma(self.bn3(self.conv3(x2)))  # (B, 128, N)
        # 全局最大池化 (沿最后一个维度 N)
        x2 = torch.max(x2, dim=2)[0]  # (B, 128)

        # Sum x1 and x2 to get x3
        x = x1 + x2  # (B, 128)
        # Element-wise multiplication
        # x = x.unsqueeze(1).expand(-1, N, -1).reshape(n_b * N, 128)
        # x = x3 * x  # (B*N, 128)
        # x = x.reshape(n_b, -1) # (B*N, 128)
        # x = self.branch_linear4(x)  # (128*N, 128)
        x = self.sigma(x)
        # To get (B, 128), we can sum over the N dimension (assuming permutation invariance)
        x = self.branch_linear5(x)
        x = self.sigma(x)
        x = self.branch_linear6(x)
        x = self.sigma(x)
        B = self.branch_linear7(x)
        return B

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        T, x3 = self.trunk_net(y)
        B = self.branch_net(C, x, x3)
        out = (T * B.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        out = torch.tanh(out)
        out = self.out_linear1(out)
        out = torch.tanh(out)
        out = self.out_linear2(out)
        return out





