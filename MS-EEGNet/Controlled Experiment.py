import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入您的数据处理模块与 MS-EEGNet
from config import CONFIG
from data_loader import EEGDataLoader
from data_processor import DataProcessor
from train import prepare_dl_data
from br_tad_engine import Orthogonal_Source_BR_TAD_Engine

try:
    from EEGNet_53 import EEGNet as MS_EEGNet
except ImportError:
    print("⚠️ 找不到 EEGNet_53.py，请确保文件在当前目录！")
    exit()

# ============================================================================
# 【核心配置区】您可以根据需要调整训练参数
# ============================================================================
BATCH_SIZE = 32
MAX_EPOCHS = 150  # 最大训练轮数
PATIENCE = 30  # 早停耐心值
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

os.makedirs('models', exist_ok=True)


# ============================================================================
# 基线模型全家桶 (结构定义 - 完整无删减)
# ============================================================================
class Conv2dWithNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.elu = nn.ELU()

    def forward(self, x): return self.elu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // ratio), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, filters, depth, kernel_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.layers.append(nn.Sequential(
                CausalConv1d(in_ch if i == 0 else filters, filters, kernel_size, dilation=dilation),
                nn.BatchNorm1d(filters), nn.ELU(),
                CausalConv1d(filters, filters, kernel_size, dilation=dilation),
                nn.BatchNorm1d(filters), nn.ELU()
            ))
        self.proj = nn.Conv1d(in_ch, filters, 1) if in_ch != filters else nn.Identity()

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            res = layer(out)
            out = res + self.proj(out) if i == 0 else res + out
        return out


class ShallowConvNet_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25))
        self.conv2 = nn.Conv2d(40, 40, (n_ch, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        with torch.no_grad():
            x = self.pool(self.bn(self.conv2(self.conv1(torch.zeros(1, 1, n_ch, n_t)))))
            self.dim = x.numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = torch.square(self.bn(self.conv2(self.conv1(x))))
        return self.fc(F.dropout(torch.log(torch.clamp(self.pool(x), min=1e-7)), 0.5).view(x.size(0), -1))


class DeepConvNet_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)), nn.Conv2d(25, 25, (n_ch, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), (1, 3)), nn.Dropout(0.5),
            Conv2dWithNorm(25, 50, (1, 10)), nn.MaxPool2d((1, 3), (1, 3)), nn.Dropout(0.5),
            Conv2dWithNorm(50, 100, (1, 10)), nn.MaxPool2d((1, 3), (1, 3)), nn.Dropout(0.5),
            Conv2dWithNorm(100, 200, (1, 10)), nn.MaxPool2d((1, 3), (1, 3)), nn.Dropout(0.5)
        )
        with torch.no_grad(): self.dim = self.net(torch.zeros(1, 1, n_ch, n_t)).numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x): return self.fc(self.net(x).view(x.size(0), -1))


class EEGNet_Standard_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t, F1=8, D=2):
        super().__init__()
        F2 = F1 * D
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False)
        self.pointwise = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        with torch.no_grad():
            x = self.pool2(self.bn3(self.pointwise(self.separable(
                self.pool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(torch.zeros(1, 1, n_ch, n_t)))))))))))
            self.dim = x.numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = self.pool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x))))))
        x = self.pool2(F.elu(self.bn3(self.pointwise(self.separable(x)))))
        return self.fc(F.dropout(x, 0.25).view(x.size(0), -1))


class EEGTCNet_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.eegnet_ext = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=8, D=2)
        self.tcn = TCNBlock(16, 12, depth=2, kernel_size=4)
        with torch.no_grad():
            x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(
                self.eegnet_ext.pool1(F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(
                    self.eegnet_ext.bn1(self.eegnet_ext.conv1(torch.zeros(1, 1, n_ch, n_t))))))))))))
            x = self.tcn(x.squeeze(2))[:, :, -1]
            self.dim = x.numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = self.eegnet_ext.pool1(
            F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(self.eegnet_ext.bn1(self.eegnet_ext.conv1(x))))))
        x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(x)))))
        x = self.tcn(x.squeeze(2))
        return self.fc(x[:, :, -1])


class TCNet_Fusion_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.eegnet_ext = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=24, D=2)
        self.tcn = TCNBlock(48, 12, depth=2, kernel_size=4)
        with torch.no_grad():
            x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(
                self.eegnet_ext.pool1(F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(
                    self.eegnet_ext.bn1(self.eegnet_ext.conv1(torch.zeros(1, 1, n_ch, n_t))))))))))))
            feat_seq = x.squeeze(2)
            tcn_out = self.tcn(feat_seq)
            concat = torch.cat([feat_seq.contiguous().view(1, -1), tcn_out.contiguous().view(1, -1)], dim=1)
            self.dim = concat.shape[1]
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = self.eegnet_ext.pool1(
            F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(self.eegnet_ext.bn1(self.eegnet_ext.conv1(x))))))
        x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(x)))))
        feat_seq = x.squeeze(2)
        tcn_out = self.tcn(feat_seq)
        concat = torch.cat([feat_seq.contiguous().view(x.size(0), -1), tcn_out.contiguous().view(x.size(0), -1)], dim=1)
        return self.fc(concat)


class ATCNet_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.eegnet_ext = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=16, D=2)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.tcn = TCNBlock(32, 32, depth=2, kernel_size=4)
        with torch.no_grad():
            x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(
                self.eegnet_ext.pool1(F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(
                    self.eegnet_ext.bn1(self.eegnet_ext.conv1(torch.zeros(1, 1, n_ch, n_t))))))))))))
            x = x.squeeze(2).transpose(1, 2)
            x, _ = self.attn(x, x, x)
            x = self.tcn(x.transpose(1, 2))
            self.dim = x[:, :, -1].numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = self.eegnet_ext.pool1(
            F.elu(self.eegnet_ext.bn2(self.eegnet_ext.depthwise(self.eegnet_ext.bn1(self.eegnet_ext.conv1(x))))))
        x = self.eegnet_ext.pool2(F.elu(self.eegnet_ext.bn3(self.eegnet_ext.pointwise(self.eegnet_ext.separable(x)))))
        x = x.squeeze(2).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = self.tcn(x.transpose(1, 2))
        return self.fc(x[:, :, -1])


class MBEEG_SENet_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.b1 = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=4, D=2)
        self.b2 = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=8, D=2)
        self.b3 = EEGNet_Standard_PT(n_ch, n_cls, n_t, F1=16, D=2)
        self.se1 = SEBlock(8)
        self.se2 = SEBlock(16)
        self.se3 = SEBlock(32)
        self.dim = self.b1.dim + self.b2.dim + self.b3.dim
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        o1 = self.b1.pool2(F.elu(self.b1.bn3(self.b1.pointwise(
            self.b1.separable(self.b1.pool1(F.elu(self.b1.bn2(self.b1.depthwise(self.b1.bn1(self.b1.conv1(x)))))))))))
        o2 = self.b2.pool2(F.elu(self.b2.bn3(self.b2.pointwise(
            self.b2.separable(self.b2.pool1(F.elu(self.b2.bn2(self.b2.depthwise(self.b2.bn1(self.b2.conv1(x)))))))))))
        o3 = self.b3.pool2(F.elu(self.b3.bn3(self.b3.pointwise(
            self.b3.separable(self.b3.pool1(F.elu(self.b3.bn2(self.b3.depthwise(self.b3.bn1(self.b3.conv1(x)))))))))))
        out = torch.cat(
            [self.se1(o1).view(x.size(0), -1), self.se2(o2).view(x.size(0), -1), self.se3(o3).view(x.size(0), -1)],
            dim=1)
        return self.fc(out)


class EEGNeX_PT(nn.Module):
    def __init__(self, n_ch, n_cls, n_t):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False)
        self.conv2 = nn.Conv2d(8, 32, (1, 32), padding=(0, 16), bias=False)
        self.dw = nn.Conv2d(32, 64, (n_ch, 1), groups=32, bias=False)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.conv3 = nn.Conv2d(64, 32, (1, 16), dilation=(1, 2), padding=(0, 15), bias=False)
        self.conv4 = nn.Conv2d(32, 8, (1, 16), dilation=(1, 4), padding=(0, 30), bias=False)
        with torch.no_grad():
            x = self.conv4(self.conv3(self.pool1(self.dw(self.conv2(self.conv1(torch.zeros(1, 1, n_ch, n_t)))))))
            self.dim = x.numel()
        self.fc = nn.Linear(self.dim, n_cls)

    def forward(self, x):
        x = F.elu(self.conv2(F.elu(self.conv1(x))))
        x = F.dropout(self.pool1(F.elu(self.dw(x))), 0.5)
        x = F.dropout(F.elu(self.conv4(F.elu(self.conv3(x)))), 0.5)
        return self.fc(x.view(x.size(0), -1))



def load_and_prepare_dual_data():
    print(" [1/3] 正在加载硬盘上的真实 EEG 数据...")
    data_loader = EEGDataLoader(CONFIG)
    raw_samples = data_loader.load_eeg_data()
    data_processor = DataProcessor(CONFIG)
    processed_samples = data_processor.preprocess_data(raw_samples)
    processed_samples = data_processor.create_epochs(processed_samples)
    full_data, full_sessions, full_labels = data_processor.getdate(processed_samples)

    print("⚙ [2/3] 执行特征空间截断与 BR-TAD 解耦净化...")
    nyq = 0.5 * 500
    b, a = butter(4, [8.0 / nyq, 35.0 / nyq], btype='band')
    br_tad_cfg = CONFIG.get('br_tad_config', {})
    all_ch_names = processed_samples[0]['epochs_processed'].ch_names
    center_channels = br_tad_cfg['center_channels']
    center_indices = [all_ch_names.index(ch) for ch in center_channels]
    br_tad_cfg['all_channels'] = all_ch_names

    # A路：Raw 数据 (10-ch)
    X_raw_filtered = filtfilt(b, a, full_data, axis=-1).astype(np.float32)
    X_raw_10ch = X_raw_filtered[:, center_indices, :]

    # B路：Clean 数据 (10-ch)
    engine = Orthogonal_Source_BR_TAD_Engine(br_tad_cfg)
    _, X_clean, _ = engine.process_all(full_data)
    X_clean_filtered = filtfilt(b, a, X_clean, axis=-1).astype(np.float32)
    X_clean_10ch = X_clean_filtered[:, center_indices, :]

    print("[3/3] 严格重置种子并划分双路数据集...")
    np.random.seed(42);
    torch.manual_seed(42);
    random.seed(42)
    splits_raw = prepare_dl_data(X_raw_10ch, full_sessions, full_labels, copy.deepcopy(CONFIG["data_selection"]))

    np.random.seed(42);
    torch.manual_seed(42);
    random.seed(42)
    splits_clean = prepare_dl_data(X_clean_10ch, full_sessions, full_labels, copy.deepcopy(CONFIG["data_selection"]))

    return splits_raw, splits_clean


# ============================================================================
# 训练与测试引擎
# ============================================================================
def train_and_eval_model(model_class, name, data_splits, device):
    X_tr, X_va, X_te, y_tr, y_va, y_te = data_splits
    n_ch, n_times = X_tr.shape[1], X_tr.shape[2]
    n_cls = len(np.unique(y_tr))

    # 初始化模型
    if "Standard EEGNet" in name:
        model = model_class(n_ch, n_cls, n_times, F1=8, D=2).to(device)
    else:
        model = model_class(n_ch, n_cls, n_times).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 数据载入 DataLoader
    X_tr_t = torch.from_numpy(X_tr[:, np.newaxis, :, :]).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    X_va_t = torch.from_numpy(X_va[:, np.newaxis, :, :]).float().to(device)
    y_va_t = torch.from_numpy(y_va).long().to(device)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    # ---- 训练循环 ----
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                preds = model(vx).argmax(dim=1)
                val_correct += (preds == vy).sum().item()

        val_acc = val_correct / len(y_va)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            break

    # ---- 测试真实准确率 ----
    model.load_state_dict(best_weights)
    model.eval()
    X_te_t = torch.from_numpy(X_te[:, np.newaxis, :, :]).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)
    with torch.no_grad():
        test_preds = model(X_te_t).argmax(dim=1)
        final_test_acc = (test_preds == y_te_t).sum().item() / len(y_te) * 100.0

    # ---- 测速 ----
    dummy_input = torch.randn(1, 1, n_ch, n_times).to(device)
    for _ in range(20): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()

    latency = (time.perf_counter() - start_time) * 1000 / 100

    print(f"  [{name}] 测试集准确率: {final_test_acc:.2f}% | 推理延迟: {latency:.2f}ms")

    return params, final_test_acc, latency


# ============================================================================

# ============================================================================
def plot_trajectory_pareto(results):
    print("\n🎨 正在绘制架构重塑动态图 (Trajectory Pareto)...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)

    latencies_all = []
    accuracies_all = []

    for i, r in enumerate(results):
        name = r["name"]
        color = r["color"]

        lat_r, acc_r = r["raw"]["latency"], r["raw"]["accuracy"]
        lat_c, acc_c = r["clean"]["latency"], r["clean"]["accuracy"]

        latencies_all.extend([lat_r, lat_c])
        accuracies_all.extend([acc_r, acc_c])

        size_r = np.sqrt(r["raw"]["params"]) * 1.0
        size_c = np.sqrt(r["clean"]["params"]) * 1.0

        # 1. 虚线空心圈：Baseline (未净化)
        ax.scatter(lat_r, acc_r, s=size_r, facecolors='none', edgecolors=color,
                   linewidth=1.8, alpha=0.6, linestyle='--')

        # 2. 实心圈：Proposed (净化后)
        ax.scatter(lat_c, acc_c, s=size_c, c=color, edgecolors='black',
                   linewidth=1.2, alpha=0.8)

        # 3. 动态箭头
        radius_r = np.sqrt(size_r) / 2
        radius_c = np.sqrt(size_c) / 2

        arrow_style = "-|>"
        if acc_c < acc_r:
            # 下降：红色虚线箭头警示
            ax.annotate("", xy=(lat_c, acc_c), xytext=(lat_r, acc_r),
                        arrowprops=dict(arrowstyle=arrow_style, color='#EF5350', lw=2.5, alpha=0.8, linestyle='--',
                                        shrinkA=radius_r + 3, shrinkB=radius_c + 3))
        else:
            # 上升：常规实线箭头
            ax.annotate("", xy=(lat_c, acc_c), xytext=(lat_r, acc_r),
                        arrowprops=dict(arrowstyle=arrow_style, color=color, lw=2.5, alpha=0.8,
                                        shrinkA=radius_r + 3, shrinkB=radius_c + 3))

        # 4. 标签定位
        if "Proposed" in name:

            ax.scatter(lat_c, acc_c, s=size_c * 1.5, c='none', edgecolors='red', linewidth=2, linestyle='--')
            ax.annotate(name, (lat_c, acc_c),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='#D32F2F')
        else:

            ax.annotate(name, (lat_c, acc_c),
                        xytext=(5, -5),
                        textcoords='offset points',
                        ha='left', va='top',
                        fontsize=9, alpha=0.9)

    ax.set_xlabel('Single-Trial Inference Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=13, fontweight='bold')

    # 自动坐标范围缩放
    if latencies_all and accuracies_all:
        ax.set_xlim(min(latencies_all) * 0.5, max(latencies_all) * 1.1)
        ax.set_ylim(min(accuracies_all) * 0.95, 100.5)

    legend_elements = [
        ax.scatter([], [], s=120, facecolors='none', edgecolors='black', linestyle='--', linewidth=1.5,
                   label='Baseline (10-ch Unpurified)'),
        ax.scatter([], [], s=120, c='gray', edgecolors='black', linewidth=1.2,
                   label='Proposed (10-ch Purified by BR-TAD)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./Pareto_Trajectory_Shift_Refined.png")
    print("✅ 图 8 已保存至: ./Pareto_Trajectory_Shift_Refined.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用 {device} 启动双路全自动训练流水线\n")

    # 1. 准备双版本数据
    splits_raw, splits_clean = load_and_prepare_dual_data()

    # 2. 模型库与配色
    MODELS = {
        "ShallowConvNet": ShallowConvNet_PT,
        "DeepConvNet": DeepConvNet_PT,
        "Standard EEGNet": EEGNet_Standard_PT,
        "ATCNet": ATCNet_PT,
        "TCNet_Fusion": TCNet_Fusion_PT,
        "EEGTCNet": EEGTCNet_PT,
        "MBEEG_SENet": MBEEG_SENet_PT,
        "EEGNeX": EEGNeX_PT,
        "MS-EEGNet": MS_EEGNet
    }

    COLOR_MAP = {
        "ShallowConvNet": '#90CAF9', "DeepConvNet": '#FFA726', "Standard EEGNet": '#29B6F6',
        "ATCNet": '#AB47BC', "TCNet_Fusion": '#26A69A', "EEGTCNet": '#5C6BC0',
        "MBEEG_SENet": '#8D6E63', "EEGNeX": '#78909C', "MS-EEGNet": '#D32F2F'
    }

    # 3. 遍历双路训练
    results = []
    for name, cls in MODELS.items():
        print(f"\n=============================================")
        print(f"🔄 模型: {name}")

        # 跑 A 路
        params_raw, acc_raw, lat_raw = train_and_eval_model(cls, name + " (Without BR-TAD)", splits_raw, device)

        # 跑 B 路
        params_clean, acc_clean, lat_clean = train_and_eval_model(cls, name + " (With BR-TAD)", splits_clean, device)

        results.append({
            "name": name,
            "color": COLOR_MAP[name],
            "raw": {"latency": lat_raw, "accuracy": acc_raw, "params": params_raw},
            "clean": {"latency": lat_clean, "accuracy": acc_clean, "params": params_clean}
        })

    # 4. 生成拉升/跌落图
    plot_trajectory_pareto(results)