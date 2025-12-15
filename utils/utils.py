import numpy as np
from torch.utils import data
import torch
import os
import random
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
def file_save_check(file):
    make_file('./Plot')
    make_file('./Result/loss_curve')
    make_file(file)
def make_file(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
def setup_seed(seed):
    # torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.backends.cudnn.deterministic = True
class DataGenerator(data.Dataset):
    def __init__(self, s, y, u, batch_size=64):
        'Initialization'
        self.s = torch.Tensor(s)  # input sample
        self.y = torch.Tensor(y)  # location
        self.u = torch.Tensor(u)  # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)

        self.N = s.shape[0]
        self.batch_size = batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        'Generates data containing batch_size samples'
        idx = np.random.choice(self.N, (self.batch_size,), replace=False)
        u = self.u[idx, :]
        y = self.y[idx, :]
        s = self.s[idx, :]
        # Construct batch
        inputs = (s, y)
        outputs = u
        return inputs, outputs

def safe_r2_score(y_true, y_pred, method='auto', verbose=True, eps=1e-8):
    """
    Compute a more stable R² score when true values are very small.

    Parameters:
    - y_true: np.ndarray
    - y_pred: np.ndarray
    - method: 'auto', 'standardize', 'scale', or 'mape'
    - verbose: whether to print diagnostics
    - eps: threshold for detecting 'small variance' in y_true

    Returns:
    - score: R² or fallback metric
    - info: dict with method used and stats
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    std_y = np.std(y_true)
    var_y = np.var(y_true)

    # if verbose:
    #     print(f"🔍 y_true std: {std_y:.4e}, var: {var_y:.4e}")

    use_alt = (var_y < eps)

    info = {'std_y': std_y, 'method_used': None}

    if method == 'standardize' or (method == 'auto' and use_alt):
        # Standardize both arrays
        scaler = StandardScaler()
        y_true_scaled = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_scaled = scaler.transform(y_pred.reshape(-1, 1)).flatten()
        score = r2_score(y_true_scaled, y_pred_scaled)
        info['method_used'] = 'standardized_r2'
        print(f"✅ Using method: {info['method_used']}, Score: {score:.6f}")


    elif method == 'scale' or (method == 'auto' and use_alt):
        # Multiply both arrays by a constant
        scale_factor = 1e3
        score = r2_score(y_true * scale_factor, y_pred * scale_factor)
        info['method_used'] = f'scaled_r2 (×{scale_factor})'

    elif method == 'mape' or (method == 'auto' and use_alt):
        score = -mean_absolute_percentage_error(y_true + eps, y_pred + eps)  # negative for consistency
        info['method_used'] = 'negative_mape'

    else:
        # Use regular R²
        score = r2_score(y_true, y_pred)
        info['method_used'] = 'standard_r2'

    # if verbose:
    #     print(f"✅ Using method: {info['method_used']}, Score: {score:.6f}")

    return score, info
