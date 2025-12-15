import torch
import matplotlib.pyplot as plt
import sys
import random
import torch.nn.functional as F
import pyvista as pv
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.pointOnet import PointONet3D_multi_u
from model.deepOnet import DeepONet3D_multi_u, DeepONet3D_multi_u_v2
from utils.utils import DataGenerator, file_save_check, safe_r2_score
from model.pointnet import PointNet3D_multi_u


class Solver_rec(object):
    def __init__(self, model_name, num_node = 40000, train_num = 2400, test='v1'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_out = 4
        self.batchsize = 16
        # self.batchsize = 10
        self.num_node = num_node
        self.train_num = train_num
        # self.epochs = 500
        self.epochs = 2000
        # self.epochs = 2
        if model_name == 'DeepONet3D':
            self.net = DeepONet3D_multi_u().to(self.device)
            # self.net = DeepONet3D_multi_u_v2().to(self.device)
        elif model_name == 'PointNet3D':
            self.net = PointNet3D_multi_u().to(self.device)
        elif model_name == 'PointONet3D':
            self.net = PointONet3D_multi_u().to(self.device)

        self.model_name = model_name + test
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)
        self.loss_log = []
        # self.criterion = F.l1_loss
        self.criterion = F.mse_loss
        self.s_tr, self.y_tr, self.u_tr, self.min_s, self.max_s, self.min_y, self.max_y, self.min_u, self.max_u = (
            self.resample_to_fixed_size(is_train_test='train', num_node=self.num_node))
        # self.s_te, self.y_te, self.u_te, _, _, _, _, _, _ = self.resample_to_fixed_size(
        #     train_test='test', num_node=self.num_node*10)
        self.s_te, self.y_te, self.u_te, _, _, _, _, _, _ = self.resample_to_fixed_size(
            is_train_test='test', num_node=self.num_node)
        self.train_loader, self.test_loader = (self.set_dataset(self.s_tr, self.y_tr, self.u_tr, self.batchsize),
                                              self.set_dataset(self.s_te, self.y_te, self.u_te, 300, shuffle=False))
        self.file_dir = f'{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}'
        self.save_dir = f'./Result/' + self.file_dir
        self.tb_dir = self.save_dir + '/logs'
        file_save_check(f'./Plot/' + self.file_dir)
        file_save_check(self.save_dir)
        self.train_loss_end = 0
    def train(self):
        if not os.path.exists('./Result/' +self.file_dir+f'/testloss_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}.txt'):
            print(f"Train {self.file_dir}")
            tb_writer = SummaryWriter(self.tb_dir)
            pbar = trange(self.epochs, desc='Epochs', disable=False)
            self.net.train()
            for epoch in pbar:
                train_loss, train_num = 0., 0.
                for i, (s, y, u) in enumerate(self.train_loader):
                    s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
                    u_pred = self.net(s, y)
                    loss = self.criterion(u_pred, u)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    train_loss += loss.item() * s.shape[0]
                    train_num += s.shape[0]
                self.scheduler.step()
                train_loss = train_loss / train_num
                test1, test2, test3, test4 = self.record_test_error(self.s_te, self.y_te, self.u_te)
                self.loss_log.append([train_loss, test1, test2, test3, test4])
                pbar.set_postfix({'Tr Loss': train_loss})
                tb_writer.add_scalar('Tr Loss', train_loss, epoch)
                if (epoch + 1) % 1000 == 0:
                    self.save()
            self.train_loss_end = train_loss
            self.plot_loss()

    def resample_to_fixed_size(self, is_train_test='train', num_node=200000):
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
        targets_save_path = parent_path + f'/data/Bracket_3d/{is_train_test}_data_node{num_node}.npz'
        if not os.path.exists(targets_save_path):
            matching_keys_tr = np.load(parent_path + '/data/Bracket_3d/combined_3000_split_random_train_valid.npz')[
                'train'].tolist()
            matching_keys_te = np.load(parent_path + '/data/Bracket_3d/combined_3000_split_random_train_valid.npz')[
                'valid'].tolist()
            matching_keys_tr = matching_keys_tr + matching_keys_te[:300]
            matching_keys_te = matching_keys_te[300:]
            # random.shuffle(matching_keys_tr)
            xyzdml = np.load(parent_path + '/data/Bracket_3d/xyzdmlc.npz')
            targets = np.load(parent_path + '/data/Bracket_3d/targets.npz')
            if is_train_test == 'train':
                matching_keys = matching_keys_tr
            else:
                matching_keys = matching_keys_te
            y, u, s = [], [], []
            names = []
            labels = []
            for name in matching_keys:
                current_target = targets[name]  # Shape varies: (270276,9), (179316,...), etc.
                current_xyzdml = xyzdml[name]
                N_total = current_target.shape[0]  # Get first dimension size
                if N_total>=num_node:
                    node_selected = np.random.choice(N_total, num_node, replace=False)
                else:
                    node_selected = np.random.choice(N_total, num_node, replace=True)
                y.append(current_xyzdml[node_selected, :4])
                s.append(current_xyzdml[0, -5:])
                u.append(current_target[node_selected, :])
                names.append(name)
                label_ = name.split('_')[0]
                labels.append(label_)
            s, y, u = np.array(s), np.array(y), np.array(u)
            np.savez(targets_save_path, s=s, y=y, u=u, labels = labels)
            # u = u[:, :, self.pred_out-1:self.pred_out]
            u = u[:, :, :]
        else:
            data = np.load(targets_save_path)
            # s, y, u = data['s'], data['y'], data['u'][:, :, self.pred_out-1:self.pred_out]
            s, y, u = data['s'], data['y'], data['u'][:, :, :]
        s, y, u = torch.from_numpy(s).to(self.device), torch.from_numpy(y).to(self.device), torch.from_numpy(u).to(
            self.device)
        # u = u[:,:,:3]
        s, y, u = s[:self.train_num, :], y[:self.train_num, :, :], u[:self.train_num, :, :]
        if is_train_test == 'train':
            s, min_s, max_s = self.normalize(s)
            y, min_y, max_y = self.normalize(y)
            u, min_u, max_u = self.normalize(u)
        else:
            s, min_s, max_s = self.normalize(s, load=True, min_value=self.min_s, max_value=self.max_s)
            y, min_y, max_y = self.normalize(y, load=True, min_value=self.min_y, max_value=self.max_y)
            u, min_u, max_u = self.normalize(u, load=True, min_value=self.min_u, max_value=self.max_u)
        return s, y, u, min_s, max_s, min_y, max_y, min_u, max_u

    # 归一化到[0, 1]范围
    def normalize(self, tensor, load=False, min_value=None, max_value=None):
        if not load:
            min_value = torch.min(tensor, dim=0, keepdim=True).values
            max_value = torch.max(tensor, dim=0, keepdim=True).values
            if tensor.dim() == 3:
                min_value = torch.min(min_value, dim=1, keepdim=True).values
                max_value = torch.max(max_value, dim=1, keepdim=True).values
        else:
            min_value, max_value = min_value, max_value
        return (tensor - min_value) / (max_value - min_value + 1e-8), min_value, max_value

    def inverse_minmax(self, normalized_tensor, min_value, max_value):
        """
        将Min-Max归一化后的数据还原到原始尺度
        Args:
            normalized_tensor: 归一化后的张量
            min_value: 原始数据的最小值（形状需匹配张量）
            max_value: 原始数据的最大值（形状需匹配张量）
        """
        return normalized_tensor * (max_value - min_value) + min_value

    def set_dataset(self, s, y, u, batchsize, shuffle=True):
        dataset = TensorDataset(s, y, u)
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        return loader

    def record_test_error(self, s, y, u):
        self.net.eval()
        s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
        with torch.no_grad():
            u_pred = self.net(s, y)
        u_pred, u = (self.inverse_minmax(u_pred, self.min_u, self.max_u),
                     self.inverse_minmax(u, self.min_u, self.max_u))
        # nrmse_error1 = self.criterion(u_pred[:,:,:1], u[:,:,:1])/self.criterion(u, torch.zeros_like(u))
        mse_error1 = self.criterion(u_pred[:, :, :1], u[:, :, :1])
        mse_error2 = self.criterion(u_pred[:, :, 1:2], u[:, :, 1:2])
        mse_error3 = self.criterion(u_pred[:, :, 2:3], u[:, :, 2:3])
        mse_error4 = self.criterion(u_pred[:, :, 3:4], u[:, :, 3:4])
        return mse_error1.item(), mse_error2.item(), mse_error3.item(), mse_error4.item()

    def compute_test_error(self):
        self.net.eval()
        for pred_i in range(4):
            maxae_loss, nrmse_loss, mae_loss, test_num = 0.,0.,0., 0.
            sum_squared_errors = 0.  # SSE (残差平方和)

            all_u = []  # 存储所有真实值（用于计算 y_mean）
            max_absolute_errors = []
            r2_list = []
            for i, (s, y, u) in enumerate(self.test_loader):
                s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
                with torch.no_grad():
                    u_pred = self.net(s, y)
                u_pred, u = (self.inverse_minmax(u_pred, self.min_u, self.max_u),
                             self.inverse_minmax(u, self.min_u, self.max_u))
                u_pred, u = u_pred[:, :, pred_i:pred_i+1], u[:,:,pred_i:pred_i+1]
                batch_mae = F.l1_loss(u_pred, u)
                batch_nrmse = F.mse_loss(u_pred, u) / F.mse_loss(u, torch.zeros_like(u))
                batch_max_mae = torch.max(torch.abs(u_pred - u))
                mae_loss += batch_mae.item() * s.shape[0]
                nrmse_loss += batch_nrmse.item() * s.shape[0]
                # maxae_loss += batch_max_mae.item() * s.shape[0]
                max_absolute_errors.append(batch_max_mae.item())
                test_num += s.shape[0]
                # 计算 R² 所需统计量
                sum_squared_errors += torch.sum((u_pred - u) ** 2).item()  # SSE
                r2, info = safe_r2_score(u_pred.detach().cpu().numpy(), u.detach().cpu().numpy())
                all_u.append(u.cpu())  # 收集真实值
                r2_list.append(r2)
            mae_loss, nrmse_loss = mae_loss / test_num, nrmse_loss / test_num
            maxae_loss = sum(max_absolute_errors) / len(max_absolute_errors)
            # 计算 R²
            all_u = torch.cat(all_u)  # 合并所有真实值
            u_mean = torch.mean(all_u)  # y 的均值
            sum_squared_total = torch.sum((all_u - u_mean) ** 2).item()  # SST
            r_squared = 1 - (sum_squared_errors / sum_squared_total)  # R²
            r2_average = sum(r2_list) / len(r2_list)
            # 写入误差文件
            with open('./Result/' +self.file_dir+f'/testloss_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}.txt', 'a') as f:  # 'a'模式会自动创建不存在的文件
                f.write(f'Train loss:{self.train_loss_end}\n'
                        f'{pred_i}_test Loss: Max-AE:{maxae_loss:.6f}, '
                        f'MAE:{mae_loss:.6f}, NRMSE:{nrmse_loss:.6f}, R2:{r_squared:.6f}, R2_average:{r2_average:.6f}\n')
        return mae_loss, nrmse_loss, maxae_loss

    def record_test_distribution(self):
        self.test_loader = self.set_dataset(self.s_te, self.y_te, self.u_te, 1, shuffle=False)
        for pred_i in range(4):
            mae_errors, nrmse_errors = [], []

            for i, (s, y, u) in enumerate(self.test_loader):
                s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
                with torch.no_grad():
                    u_pred = self.net(s, y)
                u_pred, u = (self.inverse_minmax(u_pred, self.min_u, self.max_u),
                             self.inverse_minmax(u, self.min_u, self.max_u))
                u_pred, u = u_pred[:, :, pred_i:pred_i+1], u[:,:,pred_i:pred_i+1]
                batch_mae = F.l1_loss(u_pred, u)
                batch_nrmse = F.mse_loss(u_pred, u) / F.mse_loss(u, torch.zeros_like(u))
                mae_errors.append(batch_mae.item())
                nrmse_errors.append(batch_nrmse.item())
            np.savez('./Result/' + self.file_dir+f'/testloss_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}_out{pred_i}.npz',
                     mae_errors = mae_errors, nrmse_errors=nrmse_errors)
    def plot_test(self):
        for i, (s, y, u) in enumerate(self.test_loader):
            s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
            with torch.no_grad():
                u_pred = self.net(s, y)
            task = 3
            u_pred, u = self.inverse_minmax(u_pred, self.min_u, self.max_u), self.inverse_minmax(u, self.min_u,self.max_u)
            positions = y[task, :, :3].cpu().numpy()  # 获取 positions
            u_pred = u_pred[task, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取预测值
            u = u[task, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取真实值

            u_error = abs(u_pred - u)  # 计算误差

            # 创建 plotter，并设置为显示三个子图
            plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1200, 400), border=False)

            global_min, global_max = np.min(u), np.max(u)

            # 第一个子图：显示 u
            scalar_label = 'u'
            point_cloud_u = pv.PolyData(positions)
            point_cloud_u[scalar_label] = u
            plotter.subplot(0, 0)
            plotter.add_mesh(
                point_cloud_u,
                scalars=scalar_label,
                cmap='jet',
                point_size=10,
                render_points_as_spheres=True,
                show_scalar_bar=False,  # 不单独显示颜色条
                opacity=1.0,
                clim=[global_min, global_max]  # 使用全局范围
            )
            plotter.add_text("u", position="upper_edge", font_size=10)

            # 第二个子图：显示 u_pred
            scalar_label = 'u_pred'
            point_cloud_u_pred = pv.PolyData(positions)
            point_cloud_u_pred[scalar_label] = u_pred
            plotter.subplot(0, 1)
            plotter.add_mesh(
                point_cloud_u_pred,
                scalars=scalar_label,
                cmap='jet',
                point_size=10,
                render_points_as_spheres=True,
                show_scalar_bar=False,  # 不单独显示颜色条
                opacity=1.0,
                clim=[global_min, global_max]  # 使用全局范围
            )
            plotter.add_text("u_pred", position="upper_edge", font_size=10)

            # 第三个子图：显示 u_error
            scalar_label = 'u_error'
            point_cloud_u_error = pv.PolyData(positions)
            point_cloud_u_error[scalar_label] = u_error
            plotter.subplot(0, 2)
            plotter.add_mesh(
                point_cloud_u_error,
                scalars=scalar_label,
                cmap='jet',
                point_size=10,
                render_points_as_spheres=True,
                show_scalar_bar=True,  # 只在最后一个子图显示颜色条
                opacity=1.0,
                clim=[global_min, global_max]  # 使用全局范围
            )
            plotter.add_text("u_error", position="upper_edge", font_size=10)

            # 确保所有子图使用相同的颜色映射范围
            # plotter.update_scalar_bar_range([global_min, global_max])
            plotter.add_scalar_bar(
                title="Value",
                position_x=0.8,  # 水平位置
                position_y=0.2,  # 垂直位置
                width=0.2,  # 宽度
                height=0.5,  # 高度
                vertical=True,  # 是否垂直（False为水平）
            )
            plotter.show()
            plotter.screenshot(f'./Plot/{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}/output_image{i}.png')
            print("Screenshot saved as 'output_image.png'.")


    def plot_test_pred(self):
        self.pred_out, task_pred = 4, 1
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
        matching_keys_te = np.load(parent_path + '/data/Bracket_3d/combined_3000_split_random_train_valid.npz')[
                'valid'].tolist()
        matching_keys_te = matching_keys_te[300:]
        xyzdml = np.load(parent_path + '/data/Bracket_3d/xyzdmlc.npz')
        targets = np.load(parent_path + '/data/Bracket_3d/targets.npz')
        scalar_label = matching_keys_te[task_pred]
        scalar_label_split = scalar_label.split('_')
        key = scalar_label_split[1] + '_' + scalar_label_split[2]
        component_u = targets[scalar_label][:,:]
        component_y = xyzdml[scalar_label][:, :4]
        component_s = xyzdml[scalar_label][0, -5:]
        s, y, u = (torch.from_numpy(component_s).reshape(1,5).to(self.device),
                   torch.from_numpy(component_y).reshape(1,-1,4).to(self.device),
                   torch.from_numpy(component_u).reshape(1,-1,4).to(self.device))
        s, min_s, max_s = self.normalize(s, load=True, min_value=self.min_s, max_value=self.max_s)
        y, min_y, max_y = self.normalize(y, load=True, min_value=self.min_y, max_value=self.max_y)
        u, min_u, max_u = self.normalize(u, load=True, min_value=self.min_u, max_value=self.max_u)

        with torch.no_grad():
            u_pred = self.net(s, y)
        task_pred_i = 0
        u_pred, u = self.inverse_minmax(u_pred, self.min_u, self.max_u), self.inverse_minmax(u, self.min_u,self.max_u)
        u_pred = u_pred[task_pred_i, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取预测值
        u = u[task_pred_i, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取真实值

        u_error = abs(u_pred - u)  # 计算误差

        # 创建 plotter，并设置为显示三个子图
        plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1200, 400), border=False)
        global_min, global_max = np.min(u), np.max(u)
        vtk_base_path = parent_path + '/data/Bracket_3d/VolumeMesh/'
        vtk_file = os.path.join(vtk_base_path, f"{key}.vtk")
        mesh1, mesh2, mesh3 = pv.read(vtk_file), pv.read(vtk_file), pv.read(vtk_file)
        if len(mesh1.points) >= 5:
            u = np.append(u, [0] * 5)
        mesh1[scalar_label] = u
        plotter.subplot(0, 0)
        plotter.add_mesh(mesh1, scalars=scalar_label, cmap='jet',
                         point_size=10, render_points_as_spheres=True,
            show_scalar_bar=False,  # 不单独显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u", position="upper_edge", font_size=10)
        plotter.add_axes()  # Adding the axes to the plot
        camera_position = plotter.camera_position
        print("Camera Position: ", camera_position)

        if len(mesh2.points) >= 5:
            u_pred = np.append(u_pred, [0] * 5)
        mesh2[scalar_label] = u_pred
        plotter.subplot(0, 1)
        plotter.add_mesh(mesh2, scalars=scalar_label, cmap='jet',
                         point_size=10, render_points_as_spheres=True,
            show_scalar_bar=False,  # 不单独显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u_pred", position="upper_edge", font_size=10)

        if len(mesh3.points) >= 5:
            u_error = np.append(u_error, [0] * 5)
        mesh3[scalar_label] = u_error
        plotter.subplot(0, 2)
        plotter.add_mesh(mesh3, scalars=scalar_label, cmap='jet',
                         point_size=10, render_points_as_spheres=True,
            show_scalar_bar=False,  # 不单独显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u_error", position="upper_edge", font_size=10)
        # 确保所有子图使用相同的颜色映射范围
        plotter.update_scalar_bar_range([global_min, global_max])
        plotter.add_scalar_bar(
            title="Value",
            position_x=0.8,  # 水平位置
            position_y=0.2,  # 垂直位置
            width=0.2,  # 宽度
            height=0.5,  # 高度
            vertical=True,  # 是否垂直（False为水平）
        )
        plotter.show()
        plotter.screenshot(f'./Plot/{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}/case4_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}_out{self.pred_out}_pred{task_pred}.png')
        print("Screenshot saved as 'output_image.png'.")
        np.savez(
            f'./Plot/{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}/case4_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}_out{self.pred_out}_pred{task_pred}.npz',
            mesh1_points=mesh1.points,
            mesh2_points=mesh2.points,
            mesh3_points=mesh3.points,
            u=u,
            u_pred=u_pred,
            u_error=u_error,
            scalar_label=scalar_label,
            global_min=global_min,
            global_max=global_max,
            key=key
        )

    def plot_test_pred_pointclould(self):
        self.pred_out, task_pred = 4, 1
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
        matching_keys_te = np.load(parent_path + '/data/Bracket_3d/combined_3000_split_random_train_valid.npz')[
            'valid'].tolist()
        matching_keys_te = matching_keys_te[300:]
        scalar_label_ = matching_keys_te[task_pred]

        s, y, u = next(iter(self.test_loader))
        s, y, u = s.to(self.device), y.to(self.device), u.to(self.device)
        with torch.no_grad():
            u_pred = self.net(s, y)
        u_pred, u = self.inverse_minmax(u_pred, self.min_u, self.max_u), self.inverse_minmax(u, self.min_u,self.max_u)
        positions = y[task_pred, :, :3].cpu().numpy()  # 获取 positions
        u_pred = u_pred[task_pred, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取预测值
        u = u[task_pred, :, self.pred_out-1:self.pred_out].cpu().numpy()  # 获取真实值

        u_error = abs(u_pred - u)  # 计算误差
        # 创建 plotter，并设置为显示三个子图
        plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1200, 400), border=False)
        global_min, global_max = np.min(u), np.max(u)


        # 第一个子图：显示 u
        scalar_label = 'u'
        point_cloud_u = pv.PolyData(positions)
        point_cloud_u[scalar_label] = u
        # 设置缩放因子，缩短 z 轴
        scale_factors = [1.0, 1.0, 0.5]  # 保持 x 和 y 轴不变，z 轴缩短为 0.5
        # 应用缩放，修改 positions 数据
        point_cloud_u = point_cloud_u.scale(scale_factors)
        # point_cloud_u = pv.PolyData(positions)
        point_cloud_u[scalar_label] = u

        plotter.subplot(0, 0)
        plotter.add_mesh(
            point_cloud_u,
            scalars=scalar_label,
            cmap='jet',
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=False,  # 不单独显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u", position="upper_edge", font_size=10)
        plotter.add_axes()  # Adding the axes to the plot
        camera_position = plotter.camera_position
        print("Camera Position: ", camera_position)

        # 第二个子图：显示 u_pred
        scalar_label = 'u_pred'
        scale_factors = [1.0, 1.0, 1]  # 保持 x 和 y 轴不变，z 轴缩短为 0.5
        # 应用缩放，修改 positions 数据
        point_cloud_u_pred= point_cloud_u.scale(scale_factors)
        # point_cloud_u_pred = pv.PolyData(positions)
        point_cloud_u_pred[scalar_label] = u_pred
        plotter.subplot(0, 1)
        plotter.add_mesh(
            point_cloud_u_pred,
            scalars=scalar_label,
            cmap='jet',
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=False,  # 不单独显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u_pred", position="upper_edge", font_size=10)

        # 第三个子图：显示 u_error
        scalar_label = 'u_error'
        scale_factors = [1.0, 1.0, 1]  # 保持 x 和 y 轴不变，z 轴缩短为 0.5
        # 应用缩放，修改 positions 数据
        point_cloud_u_error= point_cloud_u.scale(scale_factors)
        # point_cloud_u_error = pv.PolyData(positions)
        point_cloud_u_error[scalar_label] = u_error
        plotter.subplot(0, 2)
        plotter.add_mesh(
            point_cloud_u_error,
            scalars=scalar_label,
            cmap='jet',
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=False,  # 只在最后一个子图显示颜色条
            opacity=1.0,
            clim=[global_min, global_max]  # 使用全局范围
        )
        plotter.add_text("u_error", position="upper_edge", font_size=10)

        # 确保所有子图使用相同的颜色映射范围
        plotter.update_scalar_bar_range([global_min, global_max])
        plotter.add_scalar_bar(
            title="Value",
            position_x=0.8,  # 水平位置
            position_y=0.2,  # 垂直位置
            width=0.2,  # 宽度
            height=0.5,  # 高度
            vertical=True,  # 是否垂直（False为水平）
        )
        plotter.show()
        plotter.screenshot(f'./Plot/{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}/case4_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}_out{self.pred_out}_pred{task_pred}.png')
        print("Screenshot saved as 'output_image.png'.")
        np.savez(
            f'./Plot/{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}/case4_{self.model_name}_point{self.num_node}_train{self.train_num}_epoch{self.epochs}_out{self.pred_out}_pred{task_pred}.npz',
            positions=positions,
            u=u,
            u_pred=u_pred,
            u_error=u_error,
            scalar_label=scalar_label_,
            scalar_label_u_pred='u_pred',
            scalar_label_u_error='u_error',
            global_min=global_min,
            global_max=global_max
        )

    def plot_loss(self):
        plt.figure(figsize=(6, 5))
        plt.plot(self.loss_log, lw=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(['Tr loss', 'test loss'])
        plt.tight_layout()
        plt.savefig(f'./Result/'+ self.file_dir +'/logs/'+ self.file_dir+'.png')
        np.savez(f'{self.tb_dir}/{self.model_name}_curve_loss_point{self.num_node}_train{self.train_num}_epoch{self.epochs}.npz', self.loss_log)
        plt.show()

    def save(self):
        self.net.save(f'./Result/'+ self.file_dir+ '/'+ self.file_dir +'.pth')
    def load(self):
        filename = f'./Result/'+ self.file_dir+ '/'+ self.file_dir +'.pth'
        self.net.load_state_dict(torch.load(filename))

def DeepONet_diff_trainnum_numnode():
    for train_num in [1400, 1700, 2000, 2300, 2700]:
    # for train_num in [2700]:
        # for num_node in [500, 1000, 2500, 5000, 10000]:
        for num_node in [10000]:
            solver = Solver_rec(model_name='DeepONet3D',num_node = num_node,
                                train_num = train_num, test='v1')
            solver.train()
            solver.load()
            solver.compute_test_error()


    for train_num in [2700]:
        for num_node in [500, 1000, 2500, 5000, 10000, 20000]:
            solver = Solver_rec(model_name='DeepONet3D', num_node = num_node,
                                train_num = train_num, test='v1')
            solver.train()
            solver.load()
            solver.compute_test_error()

def PointNet_diff_trainnum_numnode():
    for train_num in [1400, 1700, 2000, 2300, 2700]:
        for num_node in [10000]:
            solver = Solver_rec(model_name='PointNet3D', num_node = num_node,
                                train_num = train_num, test='v1')
            solver.train()
            # solver.load()
            solver.compute_test_error()
            # solver.plot_test()
    for train_num in [2700]:
        for num_node in [500, 1000, 2500, 5000, 10000, 20000]:
            solver = Solver_rec(model_name='PointNet3D', num_node = num_node,
                                train_num = train_num, test='v1')
            solver.train()
            # solver.load()
            solver.compute_test_error()
            # solver.plot_test()

def PoinONet_diff_trainnum_numnode():
    for train_num in [1400, 1700, 2000, 2300, 2700]:
        for num_node in [10000]:
            solver = Solver_rec(model_name='PointONet3D',num_node = num_node,
                                train_num = train_num, test='v3')
            solver.train()
            # solver.load()
            solver.compute_test_error()
            # solver.plot_test()
    # for train_num in [2700]:
    #     # for num_node in [500, 1000, 2500, 5000, 10000, 20000]:
    #     for num_node in [10000]:
    #         solver = Solver_rec(model_name='PointONet3D', num_node = num_node,
    #                             train_num = train_num, test='v3')
    #         solver.train()
    #         solver.load()
    #         solver.compute_test_error()
    #         solver.plot_test()

if __name__ == "__main__":
    # DeepONet_diff_trainnum_numnode()
    # PointNet_diff_trainnum_numnode()
    PoinONet_diff_trainnum_numnode()

