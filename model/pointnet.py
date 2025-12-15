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

class PointNet2D(nn.Module):
    def __init__(self, N, P=10000):
        super(PointNet2D, self).__init__()
        num_point_layer = 2
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, 1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)

        self.conv7 = nn.Conv1d(96, 64, 1)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(64, 1, 1)
        self.bn8 = nn.BatchNorm1d(1)

        self.trunk_linear1 = nn.Linear(3, 32)

        # 新增 trunk_net 的 Linear 层
        self.trunk_linear1 = nn.Linear(3, 32)
        self.trunk_linear2 = nn.Linear(32, 64)
        self.trunk_linear3 = nn.Linear(64, 128)
        self.trunk_final1 = nn.Linear(129, 128)
        self.trunk_final2 = nn.Linear(128, 256)
        self.trunk_final3 = nn.Linear(256, 512)
        self.trunk_final4 = nn.Linear(512, 1024)


        self.out_linear1 = nn.Linear(N, 200)
        self.out_linear2 = nn.Linear(200, P)
        self.sigma = torch.tanh
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load
    def forward(self, s, y):
        N = s.shape[2]
        x1 = s
        # 第一层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn1(self.conv1(x1)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn2(self.conv2(x1)))  # (B, 32, N)
        # 第三层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn3(self.conv3(x1)))  # (B, 32, N)

        x2 = self.sigma(self.bn4(self.conv4(x1)))
        x2 = torch.max(x2, dim=2)[0]  # (B, 512)
        x2 = x2.unsqueeze(2).expand(-1, -1, N)

        x3 = torch.concatenate([x2,x1],dim=1)
        x3 = self.sigma(self.bn7(self.conv7(x3)))
        x3 = self.sigma(self.bn8(self.conv8(x3)))
        out = x3.permute(0, 2, 1)
        out = self.out_linear1(out.squeeze(2))
        out = self.out_linear2(out)
        return out

class PointNet1D(nn.Module):
    def __init__(self, N, P=10000):
        super(PointNet1D, self).__init__()
        num_point_layer = 2
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, 1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)

        self.conv7 = nn.Conv1d(96, 64, 1)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(64, 1, 1)
        self.bn8 = nn.BatchNorm1d(1)

        self.trunk_linear1 = nn.Linear(3, 32)

        # 新增 trunk_net 的 Linear 层
        self.trunk_linear1 = nn.Linear(3, 32)
        self.trunk_linear2 = nn.Linear(32, 64)
        self.trunk_linear3 = nn.Linear(64, 128)
        self.trunk_final1 = nn.Linear(129, 128)
        self.trunk_final2 = nn.Linear(128, 256)
        self.trunk_final3 = nn.Linear(256, 512)
        self.trunk_final4 = nn.Linear(512, 1024)


        self.out_linear1 = nn.Linear(N, 200)
        self.out_linear2 = nn.Linear(200, P)
        self.sigma = torch.tanh
        self.sigma2 = torch.tanh
        self.sigma3 = torch.tanh

        def save(filepath):
            torch.save(self.state_dict(), filepath)
        self.save = save
        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load
    def forward(self, s, y):
        N = s.shape[2]
        x1 = s
        # 第一层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn1(self.conv1(x1)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        # x1 = self.sigma(self.bn2(self.conv2(x1)))  # (B, 32, N)
        # # 第三层: Conv1d + BN + SiLU
        # x1 = self.sigma(self.bn3(self.conv3(x1)))  # (B, 32, N)

        x2 = self.sigma(self.bn4(self.conv4(x1)))
        x2 = torch.max(x2, dim=2)[0]  # (B, 512)
        x2 = x2.unsqueeze(2).expand(-1, -1, N)

        x3 = torch.concatenate([x2,x1],dim=1)
        x3 = self.sigma(self.bn7(self.conv7(x3)))
        x3 = self.sigma(self.bn8(self.conv8(x3)))
        out = x3.permute(0, 2, 1)
        out = self.out_linear1(out.squeeze(2))
        out = self.out_linear2(out)
        return out

class PointNet2D_v1(nn.Module):
    def __init__(self, loc_branch_neuron, branch_net_channel, trunk_net):
        super(PointNet2D_v1, self).__init__()
        self.branch_net = Pointlayer(neuron=loc_branch_neuron, channel=branch_net_channel)
        self.out_linear1 = MLP([loc_branch_neuron, loc_branch_neuron, 1])

        def save(filepath):
            torch.save(self.state_dict(), filepath)

        self.save = save

        def load(filename):
            self.load_state_dict(torch.load(filename))

        self.load = load

    def forward(self, s, y):
        B = self.branch_net(s)
        out = self.out_linear1(B)
        return out


class PointNet3D_multi_u(nn.Module):
    def __init__(self, ):
        super(PointNet3D_multi_u, self).__init__()
        n_c = 5
        num_point_layer = 8
        self.conv1 = nn.Conv1d(num_point_layer, 32, 1)  # (B, 32, N)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 1)  # (B, 64, N)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, 1)  # (B, 128, N)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, 1)  # (B, 128, N)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 256, 1)  # (B, 128, N)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 512, 1)  # (B, 128, N)
        self.bn6 = nn.BatchNorm1d(512)

        self.conv7 = nn.Conv1d(544, 256, 1)  # (B, 128, N)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 128, 1)  # (B, 128, N)
        self.bn8 = nn.BatchNorm1d(128)
        self.conv9 = nn.Conv1d(128, 64, 1)  # (B, 128, N)
        self.bn9 = nn.BatchNorm1d(64)
        self.conv10 = nn.Conv1d(64, 4, 1)  # (B, 128, N)
        self.bn10 = nn.BatchNorm1d(4)


        self.trunk_linear1 = nn.Linear(3, 32)

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

        def load(filename):
            self.load_state_dict(torch.load(filename))
        self.load = load

    def forward(self, C, X):
        y, x = X[:, :, :4], X[:,:, :3]
        N = X.shape[1]
        C = C.unsqueeze(1).expand(-1, N, -1)
        x1 = torch.concatenate([x,C],dim=2)
        x1 = x1.permute(0, 2, 1)  # (B, 8, N)

        # 第一层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn1(self.conv1(x1)))  # (B, 32, N)
        # 第二层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn2(self.conv2(x1)))  # (B, 32, N)
        # 第三层: Conv1d + BN + SiLU
        x1 = self.sigma(self.bn3(self.conv3(x1)))  # (B, 32, N)

        x2 = self.sigma(self.bn4(self.conv4(x1)))
        x2 = self.sigma(self.bn5(self.conv5(x2)))
        x2 = self.sigma(self.bn6(self.conv6(x2)))
        x2 = torch.max(x2, dim=2)[0]  # (B, 512)
        x2 = x2.unsqueeze(2).expand(-1, -1, N)

        x3 = torch.concatenate([x2,x1],dim=1)
        x3 = self.sigma(self.bn7(self.conv7(x3)))
        x3 = self.sigma(self.bn8(self.conv8(x3)))
        x3 = self.sigma(self.bn9(self.conv9(x3)))
        x3 = self.sigma(self.bn10(self.conv10(x3)))
        out = x3.permute(0, 2, 1)  # (B, N, 4)
        return out
