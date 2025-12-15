import numpy as np
import torch
import jax
from tqdm import trange
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.pointOnet import PointONet2D, PointONet2D_separate
from model.deepOnet import DeepONet2D
from model.pointnet import PointNet2D
from utils.utils import DataGenerator, file_save_check, setup_seed, safe_r2_score
from data.Burger_2d.dataset import generate_training_data, generate_test_data, generate_test_data_uniform
from data.Burger_2d.dataset_pointnet import generate_data_pointnet

def data_prepare(N_train=5000, m=100):
    N, P = N_train, 100
    batch_size = 10000 # batch，批次数量
    parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
    targets_save_path = parent_path + f'/data/Burger_2d/train_data_N{N_train}_m{m}.npz'
    if not os.path.exists(targets_save_path):
        print("Generating training set:")
        s_x, s, y, u = generate_training_data(N, P, m)
        np.savez(targets_save_path, s_x=s_x, s=s, y=y, u=u)
    else:
        print("Loading training set:")
        data = np.load(targets_save_path)
        s_x, s, y, u = data['s_x'], data['s'], data['y'], data['u']
    p_dataset, dataset = (DataGenerator(s_x, y, u, batch_size),
                          DataGenerator(s, y, u, batch_size))
    test_key = jax.random.PRNGKey(3)
    s_x, s, y, u = generate_data_pointnet(test_key, N_train, P, m)
    pointnet_dataset = DataGenerator(s_x, y, u, batch_size=100)
    return p_dataset, dataset, pointnet_dataset

class Solver_rec(object):
    def __init__(self, model_name, N_train, m):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file_save_check('./')
        self.model_name, self.N_train, self.m = model_name, N_train, m
        self.epochs = 120000
        # self.epochs = 10
        pointonet_dataset, deeponet_dataset, pointnet_dataset = data_prepare(N_train, m)
        if self.model_name == 'PointONet2D':
            self.net = PointONet2D(loc_branch_neuron=50, branch_net_channel=2,
                                   trunk_net=[2, 50, 50, 50, 50]).to(self.device)
            # self.net = PointONet2D_separate(value_branch_net=[m, 100, 100], loc_branch_neuron =100, loc_branch_net_channel=1,
            #                           trunk_net=[1, 100, 100]).to(self.device)
            self.dataset = pointonet_dataset
        elif model_name == 'DeepONet2D':
            self.net = DeepONet2D(branch_net=[m, 50, 50, 50, 50],
                                  trunk_net=[2, 50, 50, 50, 50]).to(self.device)
            self.dataset = deeponet_dataset
        elif model_name == 'PointNet2D':
            self.net = PointNet2D(N=self.m).to(self.device)
            self.dataset = pointnet_dataset
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []

    def train(self):
        s_val, y_val, u_val = self.valiada_data()
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
                # self.loss_log.append([loss.detach().cpu().numpy()])
                self.loss_log.append([loss.detach().cpu().numpy(), self.record_test_error(s_val, y_val, u_val)])
                pbar.set_postfix({'Loss': loss_value})
        self.plot_loss()
        self.save()
    def valiada_data(self):
        P_test = 10000
        N_test_fun_u = 100  # 测试集函数个数
        test_key = jax.random.PRNGKey(1)
        if self.model_name == 'PointONet2D':
            s_x, s, y, u = generate_test_data(test_key, N_test_fun_u, P_test, self.m)
            s = s_x
        elif self.model_name == 'PointNet2D':
            s_x, s, y, u = generate_data_pointnet(test_key, N_test_fun_u, P_test, self.m)
            s = s_x
        else:
            s_x, s, y, u = generate_test_data_uniform(test_key, N_test_fun_u, P_test, self.m)
        return s, y, u
    def record_test_error(self, s, y, u):
        s, y = torch.Tensor(s), torch.Tensor(y)
        s, y = s.to(self.device), y.to(self.device)
        u_pred = self.net(s, y).detach().cpu().numpy()
        mse = mean_squared_error(u_pred, u)
        return mse

    def compute_test_error(self, load_model=False):
        np.random.seed(1)
        if load_model:
            self.net.load(f'./Result/{self.model_name}_obsnum{self.m}_snum{self.N_train}_epoch{self.epochs}.pth')
            print(f'Loading \n {self.model_name}')
        P_test = 10000
        nrmse_history_p, mae_list, r2_list, mse_list = [], [], [], []
        N_test_fun_u = 100  # 测试集函数个数
        for i in range(N_test_fun_u):
            test_key = jax.random.PRNGKey(i)
            if self.model_name == 'PointONet2D':
                s_x, s, y, u = generate_test_data(test_key, 1, P_test, self.m)
                s = s_x
            elif self.model_name == 'PointNet2D':
                s_x, s, y, u = generate_data_pointnet(test_key, 1, P_test, self.m)
                s = s_x
            else:
                s_x, s, y, u = generate_test_data_uniform(test_key, 1, P_test, self.m)
            s, y = torch.Tensor(s), torch.Tensor(y)
            s, y = s.to(self.device), y.to(self.device)
            u_pred = self.net(s, y).detach().cpu().numpy()

            x = np.linspace(-1 * np.pi, 1 * np.pi, 100)
            t = np.linspace(0, 5, 100)
            y = y.detach().cpu().numpy()
            XX, TT = np.meshgrid(x, t)
            U_pred = griddata(y, u_pred.flatten(), (XX, TT), method='cubic')
            U = griddata(y, u.flatten(), (XX, TT), method='cubic')
            P_error = np.linalg.norm(U_pred - U, 2) / np.linalg.norm(U, 2)
            mae = mean_absolute_error(U_pred, U)
            mse = mean_squared_error(U_pred, U)
            # r2 = r2_score(U_pred, U)
            r2, info = safe_r2_score(U_pred, U)
            mae_list.append(mae)
            r2_list.append(r2)
            mse_list.append(mse)
            nrmse_history_p.append(P_error)

            if load_model and (i == 2 or i == 5 or i == 10):
                if load_model and (self.model_name == 'PointONet2D'):
                    self.plot_s(s_x, XX, TT, U_pred, U, i)
                self.plot_pred(i, XX, TT, U_pred, U)

        nrmse_average = sum(nrmse_history_p) / len(nrmse_history_p)
        mae_average = sum(mae_list) / len(mae_list)
        r2_average = sum(r2_list) / len(r2_list)
        mse_average = sum(mse_list) / len(mse_list)

        file_path = f"./Result/{self.model_name}_errorlog_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.txt"  # 文件路径
        with open(file_path, 'w') as file:  # 'w'模式覆盖写入，'a'模式追加
            # 可选：保存其他信息（如时间戳、数据维度等）
            file.write(f"NRMSE num_train{N_train}: {nrmse_average}\n")
            file.write(f"mae num_train{N_train}: {mae_average}\n")
            file.write(f"r**2 num_train{N_train}: {r2_average}\n")
        return mse_average

    def plot_pred(self, i, XX,TT,S_pred, S_test):
        fig = plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.pcolor(XX, TT, S_test, cmap='seismic')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Exact $s(x,t)$')
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.pcolor(XX, TT, S_pred, cmap='seismic')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Predict $s(x,t)$')
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.pcolor(XX, TT, abs(S_pred - S_test), cmap='seismic')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.savefig(f'./Plot/{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}_compare_{i}.pdf')
        np.savez(
            f'./save_plot/case3_{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}_compare_{i}.npz',
            u_exact=S_test, u_pred=S_pred, XX=XX, YY=TT)

    def plot_s(self, u_p, XX, TT, S_pred, S_test, i):
        x = u_p[1, 0, :]
        y = u_p[1, 1, :]
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        interpolation_func = interp1d(x_sorted, y_sorted, kind='cubic')  # 选择样条插值
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), 1000)  # 生成新的x值
        y_new = interpolation_func(x_new)  # 得到插值后的y值
        fontsizes = 18
        fig = plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(x_new, y_new, label='Ground Truth')
        plt.scatter(u_p[1, 0, :], u_p[1, 1, :], c='r', label='Measurements', marker='*')
        plt.title('s', fontsize=fontsizes)
        plt.xlabel('x', fontsize=fontsizes)
        plt.ylabel('s(x)', fontsize=fontsizes)
        plt.legend(fontsize=fontsizes)  # 显示图例
        plt.title('$s(x)$', fontsize=fontsizes)
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.pcolor(XX, TT, S_pred, cmap='seismic')
        plt.colorbar()
        plt.xlabel('$x$', fontsize=fontsizes)
        plt.ylabel('$t$', fontsize=fontsizes)
        plt.title('Pred $u(x,t)$', fontsize=fontsizes)
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.pcolor(XX, TT, abs(S_pred - S_test), cmap='seismic')
        plt.colorbar()
        plt.xlabel('$x$', fontsize=fontsizes)
        plt.ylabel('$t$', fontsize=fontsizes)
        plt.title('Absolute error', fontsize=fontsizes)
        plt.tight_layout()
        # plt.savefig(f'./save_plot/{self.model_name}_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}_diff_obs_pred_{i}.png')
        np.savez(f'./save_plot/{self.model_name}_pred_diff_obs_pred_{i}.npz', x_exact=x_new, s_exact=y_new, u_p_0=u_p[1, 0, :],
                 u_p_1=u_p[1, 1, :],
                 XX=XX, TT=TT, S_pred=S_pred, S_error=abs(S_pred - S_test))
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(6, 5))
        plt.plot(self.loss_log, lw=2, label=f'{self.model_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./Result/loss_curve/case3_{self.model_name}_curve_loss_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.png')
        np.savez(f'./Result/loss_curve/case3_{self.model_name}_curve_loss_obsnum{self.m}_trainnum{self.N_train}_epoch{self.epochs}.npz', loss =self.loss_log)

    def save(self):
        self.net.save(f'./Result/{self.model_name}_obsnum{self.m}_snum{self.N_train}_epoch{self.epochs}.pth')
    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

def diff_obs_train_num():
    global m, N_train
    # for obs_num in [25, 50, 100, 125, 150]:
    for N_train in [1000]:
        for obs_num in [50]:
            np.random.seed(1)
            m = obs_num
            N_train = N_train
            # solver = Solver_rec(model_name='PointONet2D', N_train=N_train, m=m)
            # solver = Solver_rec(model_name='DeepONet2D', N_train=N_train, m=m)
            solver = Solver_rec(model_name='PointNet2D', N_train=N_train, m=m)
            # solver.train()
            solver.compute_test_error(load_model=True)

            solver = Solver_rec(model_name='PointONet2D', N_train=N_train, m=m)
            # solver.train()
            solver.compute_test_error(load_model=True)

            solver = Solver_rec(model_name='DeepONet2D', N_train=N_train, m=m)
            # solver.train()
            solver.compute_test_error(load_model=True)


if __name__ == '__main__':
    diff_obs_train_num()

