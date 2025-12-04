import os
import csv
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from astropy.io import fits
import torch.hub
import pandas as pd

# -----------------------------
# 0) 配置
# -----------------------------
base_dir    = "./"
patchs_root = os.path.join(base_dir, "patches_7_32_noise_all_0.06")   # ← 与训练一致
label_file  = os.path.join(patchs_root, "params_test.txt")            # ← 你的测试标签
weight_path = "./pth/blackhole_resnet152_7_fianl_all_0.06.pth"        # ← 注意：应为 .pth
out_dir     = "./csv/eval_resnet152_mc_all_0.06_dropout_train"        # 输出目录（多CSV+最终聚合）
os.makedirs(out_dir, exist_ok=True)

# MC 次数（论文用 ~1000；根据算力自行调整）
T_MC = 500

K_max = 2               # 二分量（每分量4个参数，共8列）
target_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数名（每个分量4个参数）
param_names_one = ["sigma_x", "sigma_y", "theta", "I_center"]

# -----------------------------
# 工具：只启用 Dropout 的 MC 模式（BN 仍 eval）
# -----------------------------
def enable_mc_dropout_only(m: nn.Module):
    """
    先整体 eval()，再把 Dropout / Dropout2d / Dropout1d 切到 train()。
    这样 BN 等层保持冻结统计，只有随机失活生效（近似 epistemic）。
    """
    m.eval()
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout1d)):
            mod.train()

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
        # 图像（保持绝对幅值）
        img_np = fits.getdata(os.path.join(self.patch_dir, fn)).astype(np.float32)
        img = self.transform(img_np).float()

        # 从文件名解析 base_fn 与多个坐标
        base, _ = os.path.splitext(fn)
        parts = base.split('_')
        # 兼容末尾 "(...)" 标识
        if parts[-1].startswith('(') and parts[-1].endswith(')'):
            parts = parts[:-1]

        base_fn = f"{parts[0]}_{parts[1]}.fits"
        coords = []
        for i in range(2, len(parts), 2):
            x, y = int(parts[i]), int(parts[i+1])
            coords.append((x, y))

        # 取前 K_max 个；不足时 pad 0
        coords = coords[:self.K_max]
        vals = []
        for (x, y) in coords:
            sx, sy, th, Icen = self.label_dict.get((base_fn, x, y), (0.0, 0.0, 0.0, 0.0))
            vals += [sx, sy, th, Icen]
        if len(coords) < self.K_max:
            vals += [0.0] * (4 * (self.K_max - len(coords)))

        return img, torch.tensor(vals, dtype=torch.float32), fn, coords, base_fn

# -----------------------------
# 3) 模型（与训练一致）
# -----------------------------
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
# 灰度输入 + 小图：3x3 conv，去掉第一层 maxpool
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

def append_dropout(m: nn.Module, p: float):
    for name, module in list(m.named_children()):
        if len(list(module.children())) > 0:
            append_dropout(module, p)
        if isinstance(module, nn.ReLU):
            setattr(m, name, nn.Sequential(module, nn.Dropout2d(p=p)))

append_dropout(model, p=0.1)

# 线性 head：2048 → 2*4*K_max = 16
num_feat = model.fc.in_features
model.fc = nn.Linear(num_feat, 2 * 4 * K_max)

# 加载权重
state = torch.load(weight_path, map_location=device)
model.load_state_dict(state)
model.to(device)

# -----------------------------
# 4) DataLoader
# -----------------------------
test_dir = os.path.join(patchs_root, "test")
test_ds = MultiGmmPatchDataset(test_dir, label_dict, K_max=K_max, target_size=target_size)
# 这里可以设更大的 batch_size，加速推理
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# -----------------------------
# 5) 公共列名与工具
# -----------------------------
def make_colnames(prefix):
    cols = []
    for k in range(1, K_max+1):
        for name in param_names_one:
            cols.append(f"{prefix}_{name}_k{k}")
    return cols

def split_flat(vec, K_max):
    out = []
    for k in range(K_max):
        block = vec[4*k:4*(k+1)]  # [σx, σy, θ, I]
        out += list(block)
    return out

param_names_all = []
for k in range(1, K_max+1):
    for n in param_names_one:
        param_names_all.append((k, n))

D = 4 * K_max

# -----------------------------
# 6) MC 前向：重复 T_MC 次，每次写一个 CSV（pred_dp_XXXX.csv）
# -----------------------------
enable_mc_dropout_only(model)  # 冻结 BN，仅开启 Dropout
print(f"Running MC Dropout with T={T_MC}")

for t in range(T_MC):
    csv_path = os.path.join(out_dir, f"pred_dp_{t:04d}.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = (["filename"]
                  + make_colnames("gt")
                  + make_colnames("pred_mu")
                  + make_colnames("pred_sigma")
                  + make_colnames("abs_err")
                  + make_colnames("rel_err_%"))
        writer.writerow(header)

        # 用于 RMSE 累积
        rmse_acc = { (k, n): [] for (k, n) in param_names_all }

        for img, lbl, fname, coords, base_fn in tqdm(test_loader, total=len(test_loader), desc=f"MC {t+1}/{T_MC}"):
            img = img.to(device)
            with torch.no_grad():
                pred_mat = model(img).cpu().numpy()        # [B, 2D]

            # 按“最后一维”切分 mu/logvar：得到 [B, D]
            mu_mat     = pred_mat[:, :D]
            logvar_mat = pred_mat[:, D:]
            sigma_mat  = np.exp(0.5 * logvar_mat)

            # 标签 [B, D]；文件名是长度为 B 的列表
            lbl_mat = lbl.numpy()                          # [B, D]
            B       = mu_mat.shape[0]

            for b in range(B):
                mu     = mu_mat[b]                         # [D]
                sigma  = sigma_mat[b]                      # [D]
                gt_vec = lbl_mat[b]                        # [D]

                # 展平成列顺序（k1 后 k2；每个分量4个参数）
                gt_vals        = split_flat(gt_vec, K_max)     # -> [8]
                pred_mu_vals   = split_flat(mu, K_max)         # -> [8]
                pred_sig_vals  = split_flat(sigma, K_max)      # -> [8]

                gt_arr = np.array(gt_vals, dtype=np.float32)
                pm_arr = np.array(pred_mu_vals, dtype=np.float32)

                abs_err = np.abs(pm_arr - gt_arr)              # [8]
                rel_err = abs_err / (np.abs(gt_arr) + 1e-8) * 100.0

                # 注意从 batch 的文件名列表里取第 b 个
                row = ([fname[b]]
                       + [f"{v:.4f}" for v in gt_arr]
                       + [f"{v:.4f}" for v in pm_arr]
                       + [f"{v:.4f}" for v in pred_sig_vals]
                       + [f"{v:.4f}" for v in abs_err]
                       + [f"{v:.2f}" for v in rel_err])
                writer.writerow(row)

                # RMSE 累积：逐列
                idx = 0
                for kk in range(1, K_max+1):
                    for n in param_names_one:
                        rmse_acc[(kk, n)].append(float(abs_err[idx]**2))
                        idx += 1

        # 写 RMSE 行（对应 abs_err_* 的 8 列）
        rmse_vals = []
        for kk in range(1, K_max+1):
            for n in param_names_one:
                se_list = rmse_acc[(kk, n)]
                mse = np.mean(se_list) if se_list else 0.0
                rmse_vals.append(np.sqrt(mse))

        rmse_row = (["RMSE"]
                    + [""] * (8 + 8 + 8)
                    + [f"{v:.6f}" for v in rmse_vals]
                    + [""] * 8)
        writer.writerow(rmse_row)

print(f"Per-pass CSVs saved to: {out_dir}")

# -----------------------------
# 7) 聚合：读取所有 per-pass CSV，按 N(mu, sigma) 采样并求最终均值/方差
# -----------------------------
csv_files = sorted(glob.glob(os.path.join(out_dir, "pred_dp_*.csv")))
assert len(csv_files) == T_MC, f"Expected {T_MC} CSVs, got {len(csv_files)}"

df0 = pd.read_csv(csv_files[0])
data_rows = df0[df0["filename"] != "RMSE"]  # 去掉 RMSE 行
N = len(data_rows)

def make_col_index_map(df):
    cols = list(df.columns)
    def idxs(prefix):
        return [cols.index(c) for c in make_colnames(prefix)]
    return {"gt": idxs("gt"), "pm": idxs("pred_mu"), "ps": idxs("pred_sigma")}

col_idx = make_col_index_map(df0)

# 固定 ground truth（所有文件应一致）
gt_mat = data_rows.iloc[:, col_idx["gt"]].to_numpy(dtype=np.float32)     # [N, 8]

# 存 posterior 样本（T, N, 8）
post_samples = np.zeros((len(csv_files), N, 8), dtype=np.float32)

for ti, path in enumerate(tqdm(csv_files, desc="Aggregate posteriors")):
    df  = pd.read_csv(path)
    dfr = df[df["filename"] != "RMSE"]
    pm  = dfr.iloc[:, col_idx["pm"]].to_numpy(dtype=np.float32)          # [N, 8]
    ps  = dfr.iloc[:, col_idx["ps"]].to_numpy(dtype=np.float32)          # [N, 8]
    # 从 N(mu, sigma) 采样（逐元素）
    sample = np.random.normal(loc=pm, scale=ps).astype(np.float32)       # [N, 8]
    post_samples[ti] = sample

# 最终均值 / 标准差（沿 MC 维聚合）
final_pred  = post_samples.mean(axis=0)                                   # [N, 8]
final_sigma = post_samples.std(axis=0, ddof=0)                            # [N, 8]

# 计算误差
abs_err_final = np.abs(final_pred - gt_mat)
rel_err_final = abs_err_final / (np.abs(gt_mat) + 1e-8) * 100.0

# 写最终 CSV（列顺序与单次 CSV 相同）
final_csv = os.path.join(out_dir, "final_pred.csv")
with open(final_csv, "w", newline='') as f:
    writer = csv.writer(f)
    header = (["filename"]
              + make_colnames("gt")
              + make_colnames("pred_mu")
              + make_colnames("pred_sigma")
              + make_colnames("abs_err")
              + make_colnames("rel_err_%"))
    writer.writerow(header)

    # 逐样本写行：沿用第一个 CSV 的文件名
    filenames = data_rows["filename"].tolist()
    for i in range(N):
        row = ([filenames[i]]
               + [f"{v:.4f}" for v in gt_mat[i]]
               + [f"{v:.4f}" for v in final_pred[i]]
               + [f"{v:.4f}" for v in final_sigma[i]]
               + [f"{v:.4f}" for v in abs_err_final[i]]
               + [f"{v:.2f}" for v in rel_err_final[i]])
        writer.writerow(row)

    # RMSE 行（与 abs_err_* 对齐，得到每个参数的最终 RMSE）
    rmse_vals = np.sqrt((abs_err_final ** 2).mean(axis=0)).tolist()
    rmse_row = (["RMSE"]
                + [""] * (8 + 8 + 8)
                + [f"{v:.6f}" for v in rmse_vals]
                + [""] * 8)
    writer.writerow(rmse_row)

print(f"Done. Final aggregated CSV saved to: {final_csv}")
