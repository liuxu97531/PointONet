import numpy as np
data = np.load('combined_3000_split_random_train_valid.npz')

print("Keys:", data.files)
data_train = data['train']
data_valid = data['valid'] # 训练集0.8，验证集0.2

targets = np.load('targets.npz') # (6315keys, (nx,4))
print("Keys:", targets.files)
draft = targets.files

label = data_train[0]


draft2 = targets[label]

mass = np.load('combined_3000_mass.npz')
draft3 = mass[label]

xyzdmlc = np.load('xyzdmlc.npz')  # (6315keys, (nx,9), 前三列为xyz，第四列为sdf，第五列为m，第六列为f，后三列为方向导数)
draft4 = xyzdmlc[label]

sorted_one_mass_data = dict(sorted(data_train.tolist()))
print(1)

