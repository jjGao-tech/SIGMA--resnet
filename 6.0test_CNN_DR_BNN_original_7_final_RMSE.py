import os
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from astropy.io import fits

# -----------------------------
# 0) 配置
# -----------------------------
base_dir    = "./"
patchs_root = os.path.join(base_dir, "patches_7_32_noise_all_0.08")  # ← 与训练一致
label_file  = os.path.join(patchs_root, "params_test.txt")
weight_path = "./pth/CNN_7_32_noise_theta_DR_BNN_all_0.08.pth"      # ← 你的CNN权重
out_dir     = "./csv"
os.makedirs(out_dir, exist_ok=True)
csv_path    = os.path.join(out_dir, "CNN_test_K2_all_0.08.csv")

K_max = 2
target_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数名（每个分量4个参数）
param_names_one = ["sigma_x", "sigma_y", "theta", "I_center"]

# -----------------------------
# 1) 读标签到字典（保持“绝对值”）
#    key: (base_fn, x, y) -> (sx, sy, th, Icen)
# -----------------------------
label_dict = {}
with open(label_file, "r") as f:
    next(f)  # 跳过表头
    for line in f:
        fn, _, _, xs, ys, _, sx, sy, th, _, Icen = line.strip().split('\t')
        key = (fn, int(xs), int(ys))
        label_dict[key] = (float(sx), float(sy), float(th), float(Icen))

# -----------------------------
# 2) 测试集 Dataset（不做 per-patch 归一化）
#    从文件名解析出 2 个 (x,y)，顺序即本样本两个分量的顺序
# -----------------------------
class MultiGmmPatchDataset(Dataset):
    def __init__(self, patch_dir, label_dict, K_max=2, target_size=32):
        self.patch_dir = patch_dir
        self.label_dict = label_dict
        self.K_max = K_max
        self.items = sorted([fn for fn in os.listdir(patch_dir) if fn.lower().endswith('.fits')])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size), antialias=True),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn = self.items[idx]
        img_np = fits.getdata(os.path.join(self.patch_dir, fn)).astype(np.float32)
        img = self.transform(img_np).float()  # [1, H, W]

        base, _ = os.path.splitext(fn)
        parts = base.split('_')
        if parts[-1].startswith('(') and parts[-1].endswith(')'):
            parts = parts[:-1]

        base_fn = f"{parts[0]}_{parts[1]}.fits"
        coords = []
        for i in range(2, len(parts), 2):
            x, y = int(parts[i]), int(parts[i+1])
            coords.append((x, y))

        coords = coords[:self.K_max]
        vals = []
        for (x, y) in coords:
            sx, sy, th, Icen = self.label_dict.get((base_fn, x, y), (0.0, 0.0, 0.0, 0.0))
            vals += [sx, sy, th, Icen]
        if len(coords) < self.K_max:
            vals += [0.0] * (4 * (self.K_max - len(coords)))

        return img, torch.tensor(vals, dtype=torch.float32), fn, coords, base_fn

# -----------------------------
# 3) CNN 模型（与训练一致）
#     输出维度 = 2 * 4 * K_max  （前一半是 mu，后一半是 logvar）
# -----------------------------
class BayesianCNN(nn.Module):
    def __init__(self, output_dim, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.Dropout2d(p=dropout_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.Dropout2d(p=dropout_rate),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_rate),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_rate),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_rate),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(1024, output_dim)

        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x):
        x = self.conv1(x)   # [B, 64, 16, 16]
        x = self.conv2(x)   # [B,128,  8,  8]
        x = self.conv3(x)   # [B,256,  4,  4]
        x = self.conv4(x)   # [B,512,  2,  2]
        x = self.conv5(x)   # [B,512,  2,  2]
        x = self.flatten(x) # [B, 2048]
        x = self.fc1(x)     # [B, 2048]
        x = self.fc2(x)     # [B, 1024]
        return self.out_layer(x)  # [B, 2*4*K_max]

# 实例化并加载权重
output_dim = 2 * 4 * K_max
model = BayesianCNN(output_dim=output_dim, dropout_rate=0.1).to(device)
state = torch.load(weight_path, map_location=device)
model.load_state_dict(state)
model.eval()

# -----------------------------
# 4) DataLoader
# -----------------------------
test_dir = os.path.join(patchs_root, "test")
test_ds = MultiGmmPatchDataset(test_dir, label_dict, K_max=K_max, target_size=target_size)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# -----------------------------
# 5) 推理 & CSV
#     列顺序：k1 后 k2；每个参数都有 k1/k2 两列
# -----------------------------
def make_colnames(prefix):
    cols = []
    for k in range(1, K_max+1):
        for name in param_names_one:
            cols.append(f"{prefix}_{name}_k{k}")
    return cols

param_names_all = []
for k in range(1, K_max+1):
    for n in param_names_one:
        param_names_all.append((k, n))  # (分量序号, 参数名)

rmse_acc = { (k, n): [] for (k, n) in param_names_all }

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)

    header = (["filename"]
              + make_colnames("gt")
              + make_colnames("pred_mu")
              + make_colnames("pred_sigma")
              + make_colnames("abs_err")
              + make_colnames("rel_err_%"))
    writer.writerow(header)

    for img, lbl, fname, coords, base_fn in tqdm(test_loader):
        img = img.to(device)
        with torch.no_grad():
            pred = model(img).cpu().squeeze(0).numpy()

        D = 4 * K_max
        mu    = pred[:D]
        logvar= pred[D:]
        sigma = np.exp(0.5 * logvar)

        def split_flat(vec):
            out = []
            for k in range(K_max):
                block = vec[4*k:4*(k+1)]  # [σx, σy, θ, I_center]
                out += list(block)
            return out

        gt_vals       = split_flat(lbl.squeeze(0).numpy())
        pred_mu_vals  = split_flat(mu)
        pred_sig_vals = split_flat(sigma)

        gt_arr = np.array(gt_vals, dtype=np.float32)
        pm_arr = np.array(pred_mu_vals, dtype=np.float32)

        abs_err = np.abs(pm_arr - gt_arr)
        rel_err = abs_err / (np.abs(gt_arr) + 1e-8) * 100.0

        row = ([fname[0]]
               + [f"{v:.4f}" for v in gt_arr]
               + [f"{v:.4f}" for v in pm_arr]
               + [f"{v:.4f}" for v in pred_sig_vals]
               + [f"{v:.4f}" for v in abs_err]
               + [f"{v:.2f}" for v in rel_err])
        writer.writerow(row)

        # RMSE 累积：逐列
        idx = 0
        for k in range(1, K_max+1):
            for n in param_names_one:
                rmse_acc[(k, n)].append(float(abs_err[idx]**2))
                idx += 1

    # 计算 RMSE（与 abs_err_* 列位置对齐）
    rmse_vals = []
    for k in range(1, K_max+1):
        for n in param_names_one:
            se_list = rmse_acc[(k, n)]
            mse = np.mean(se_list) if se_list else 0.0
            rmse_vals.append(np.sqrt(mse))

    # 对齐规则：
    # filename(1) + gt(8) + pred_mu(8) + pred_sigma(8) = 前 25 列
    # 接着 abs_err(8) —— 在这 8 列填 RMSE；最后 rel_err(8) 置空
    rmse_row = (["RMSE"]
                + [""] * (8 + 8 + 8)
                + [f"{v:.6f}" for v in rmse_vals]
                + [""] * 8)
    writer.writerow(rmse_row)

print(f"Done. CSV saved to: {csv_path}")
