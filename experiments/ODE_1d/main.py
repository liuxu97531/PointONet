import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import jax
import sys
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.pointOnet import PointONet2D,PointONet2D_separate
from model.deepOnet import DeepONet2D
from model.pointnet import PointNet2D, PointNet1D
from utils.utils import DataGenerator, file_save_check, safe_r2_score
from data.ODE_1d.dataset import generate_training_data, generate_test_data, generate_test_data_uniform_obs
from data.ODE_1d.dataset_pointnet import generate_data_pointnet

def data_prepare(N_train=10000, m=10000):
    P_train = 10
    batch_size = 10000
    key_train = jax.random.PRNGKey(0)
    parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
    targets_save_path = parent_path + f'/data/ODE_1d/train_data_N{N_train}_m{m}.npz'
    if not os.path.exists(targets_save_path):
        s_x, s, y, u, _, _, _ = generate_training_data(key_train, N_train, m, P_train)
        np.savez(targets_save_path, s_x=s_x, s=s, y=y, u=u)
    else:
        data = np.load(targets_save_path)
        s_x, s, y, u = data['s_x'], data['s'], data['y'], data['u']
    p_dataset, dataset = (DataGenerator(s_x, y, u, batch_size),
                          DataGenerator(s, y, u, batch_size))
    s_x, s, y, u = generate_data_pointnet(key_train, N_train, m, 1000)
    pointnet_dataset = DataGenerator(s_x, y, u, batch_size=100)
    return p_dataset, dataset, pointnet_dataset

class Solver_rec(object):
    def __init__(self, model_name, N_train, m, test='v1'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file_save_check('./')
        self.model_name, self.N_train, self.m = model_name, N_train, m
        self.epochs = 40000
        # self.epochs = 10
        pointonet_dataset, deeponet_dataset, pointnet_dataset = data_prepare(N_train, m)
        if self.model_name == 'PointONet2D':
            self.net = PointONet2D(loc_branch_neuron=100, branch_net_channel=2, trunk_net=[1, 100, 100]).to(self.device)
            # self.net = PointONet2D_separate(value_branch_net=[m, 100, 100], loc_branch_neuron =100, loc_branch_net_channel=1,
            #                           trunk_net=[1, 100, 100]).to(self.device)
            self.dataset = pointonet_dataset
        elif model_name == 'DeepONet2D':
            self.net = DeepONet2D(branch_net=[m, 100, 100], trunk_net=[1, 100, 100]).to(self.device)
            self.dataset = deeponet_dataset
        elif model_name == 'PointNet2D':
            # self.net = PointNet2D(N=self.m, P=1000).to(self.device)
            self.net = PointNet1D(N=self.m, P=1000).to(self.device)
            self.dataset = pointnet_dataset
        self.model_name_ = self.model_name
        self.model_name = model_name + test
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []
        self.file_dir = f'{self.model_name}_point{self.m}_train{N_train}_epoch{self.epochs}'
        self.save_dir = f'./Result/' + self.file_dir
        file_save_check(self.save_dir)

    def train(self):
        s_val, y_val, u_val = self.validate_data()
        print(f'Train the {self.model_name}')
        data_iterator = iter(self.dataset)
        pbar = trange(self.epochs)
        for it in pbar:
            self.optimizer.zero_grad()
            batch = next(data_iterator)
            (s, y), u = batch
            s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
            u_pred = self.net(s, y)
            loss = torch.mean((u.flatten() - u_pred.flatten()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            if it % 100 == 0:
                loss_value = loss.detach().cpu().numpy().item()
                self.loss_log.append([loss.detach().cpu().numpy(), self.record_test_error(s_val, y_val, u_val)])
                # self.loss_log.append(loss.detach().cpu().numpy())
                pbar.set_postfix({'Loss': loss_value})
        self.plot_loss()
        self.save()

    def validate_data(self):
        P_test = m  # number of sensors
        N_test_fun_u = 100  # 测试集函数个数
        key_test = jax.random.PRNGKey(N_test_fun_u)  # A different key
        if self.model_name_ == 'PointONet2D':
            s_x, s, y, u = generate_test_data(key_test, N_test_fun_u, m, P_test)  # 如果画测点位置可变，用这行生成数据
            s = s_x
        elif self.model_name_ == 'PointNet2D':
            P_test = 1000
            s_x, s, y, u = generate_data_pointnet(key_test, N_test_fun_u, self.m, P_test)
            s = s_x
        else:
            s_x, s, y, u = generate_test_data_uniform_obs(key_test, N_test_fun_u, m, P_test)
        return s, y, u
    def record_test_error(self,s, y, u):
        s, y = torch.Tensor(s), torch.Tensor(y)
        s, y = s.to(self.device), y.to(self.device)
        u_pred = self.net(s, y).detach().cpu().numpy()
        mse = mean_squared_error(u_pred, u)
        return mse

    def compute_test_error(self, load_model=False):
        np.random.seed(1)
        P_test = m  # number of sensors
        nrmse_history_p, mae_list, r2_list = [], [], []
        N_test_fun_u = 100  # 测试集函数个数
        if load_model:
            self.net.load('./Result/'+ self.file_dir+f'/{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.pth')
        for i in range(N_test_fun_u):
            key_test = jax.random.PRNGKey(i)  # A different key
            if self.model_name == 'PointONet2D':
                s_x, s, y, u = generate_test_data(key_test, 1, m, P_test) # 如果画测点位置可变，用这行生成数据
                s = s_x
            elif self.model_name == 'PointNet2D':
                P_test = 1000
                s_x, s, y, u = generate_data_pointnet(key_test, 1, self.m, P_test)
                s = s_x
            else:
                s_x, s, y, u = generate_test_data_uniform_obs(key_test, 1, m, P_test)
            s, y = torch.Tensor(s), torch.Tensor(y)
            s, y = s.to(self.device), y.to(self.device)
            u_pred = self.net(s, y).detach().cpu().numpy()
            nrmse = np.linalg.norm(u_pred - u, 2) / np.linalg.norm(u, 2)
            mae = mean_absolute_error(u_pred, u)
            # r2 = r2_score(u_pred, u)
            r2, info = safe_r2_score(u_pred, u)
            mae_list.append(mae)
            r2_list.append(r2)
            nrmse_history_p.append(nrmse)
            idx = 0
            index = np.arange(idx * P_test, (idx + 1) * P_test)
            # if i == 36:
            if i == 46:
                # if self.model_name=='PointONet2D' or 'PointNet2D':
                #     self.plot_s(i, index, s_x, y, u, u_pred, self.model_name)
                self.plot_pred(i, index, y, u, u_pred)

        nrmse_average = sum(nrmse_history_p) / len(nrmse_history_p)
        mae_average = sum(mae_list) / len(mae_list)
        r2_average = sum(r2_list) / len(r2_list)

        file_path = './Result/'+ self.file_dir+f'/{self.model_name}_errorlog_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.txt'  # 文件路径
        with open(file_path, 'w') as file:
            file.write(f"NRMSE num_train{N_train}: {nrmse_average}\n")
            file.write(f"mae num_train{N_train}: {mae_average}\n")
            file.write(f"r**2 num_traub{N_train}: {r2_average}\n")
    def plot_s(self, i, index, s_x, y, u, u_pred, model_name):
        y = y.detach().cpu().numpy()
        fontsizes = 18
        X = s_x[0, 0, :]
        Y = s_x[0, 1, :]
        sorted_indices = np.argsort(X)
        x_sorted = X[sorted_indices]
        y_sorted = Y[sorted_indices]
        interpolation_func = interp1d(x_sorted, y_sorted, kind='cubic')  # 选择样条插值
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), 1000)  # 生成新的x值
        y_new = interpolation_func(x_new)  # 得到插值后的y值
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(x_new, y_new, label='Exact')
        axs[0].scatter(s_x[0, 0, :], s_x[0, 1, :], c='r', label='Observations', marker='*')
        if model_name == 'PointNet2D':
            y, u, u_pred = y.T, u.T, u_pred.T
        axs[1].plot(y[index, :], u[index, :], label='Exact', lw=2)
        axs[1].plot(y[index, :], u_pred[index, :], '--', label='Pred', lw=2)
        axs[2].plot(y[index, :], abs(u[index, :] - u_pred[index, :]), label='Absolute error', lw=2)
        axs[0].set_xlabel('$x$', fontsize=fontsizes)
        axs[0].set_ylabel('$s(x)$', fontsize=fontsizes)
        axs[1].set_xlabel('$y$', fontsize=fontsizes)
        axs[1].set_ylabel('$u(y)$', fontsize=fontsizes)
        plt.tight_layout()
        axs[0].legend(fontsize=fontsizes)  # 显示图例
        # axs[1].legend(fontsize=fontsizes)  # 显示图例
        # axs[2].legend(fontsize=fontsizes)  # 显示图例
        axs[2].set_xlabel('$y$', fontsize=fontsizes)
        axs[2].set_ylim(0, 0.1)
        plt.savefig('./Result/'+ self.file_dir+f'/{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}_diff_obs_pred_{i}.png')
        plt.show()


    def plot_pred(self, i, index, y, u, u_pred):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        y = y.detach().cpu().numpy()
        if self.model_name == 'PointNet2D':
            y, u, u_pred = y.T, u.T, u_pred.T
        plt.plot(y[index, :], u[index, :], label='Exact s', lw=2)
        plt.plot(y[index, :], u_pred[index, :], '--', label=f'{self.model_name}', lw=2)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$u(y)$')
        plt.tight_layout()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(y[index, :], abs(u_pred[index, :] - u[index, :]), '--', lw=2, label='error')
        plt.tight_layout()
        plt.ylabel(r'$|u_{pred}-u|$')
        plt.ylabel(r'$u(y)$')
        plt.legend()
        plt.savefig('./Result/'+ self.file_dir+f'/Plot/{self.model_name}_pred_{i}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.png')
        np.savez(
            './Result/'+ self.file_dir+f'/case1_{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}_compare_{i}.npz',
            u_exact=u[index, :], u_pred=u_pred[index, :], XX=y[index, :], YY=y[index, :])

    def plot_loss(self):
        plt.figure(figsize=(6, 5))
        plt.plot(self.loss_log, lw=2, label=f'{self.model_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./Result/'+ self.file_dir+f'/loss_curve/case1_{self.model_name}_curve_loss_obsnum{self.m}_trainsnum{self.N_train}_epoch{self.epochs}.png')
        np.savez('./Result/'+ self.file_dir+f'/loss_curve/case1_{self.model_name}_curve_loss_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.npz', loss =self.loss_log)

    def save(self):
        self.net.save('./Result/'+ self.file_dir+f'/{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.pth')
    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

if __name__ == '__main__':
    for N_train in [1000]:
        for obs_num in [50]:
            np.random.seed(1)
            N_train = N_train # u函数个数
            m = obs_num  # number of input sensors
            solver = Solver_rec(model_name='PointONet2D', N_train=N_train, m=m, test='v3')
            solver.train()
            solver.compute_test_error(load_model=True)

            # solver = Solver_rec(model_name = 'PointONet2D', N_train=N_train, m=m)
            # # # solver.train()
            # solver.compute_test_error(load_model=True)

            # solver = Solver_rec(model_name= 'DeepONet2D', N_train=N_train, m=m, test='v3')
            # solver.train()
            # solver.compute_test_error(load_model=True)




