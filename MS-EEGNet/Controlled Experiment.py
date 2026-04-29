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
MAX_EPOCHS = 150  # 最大训练轮数（正式跑建议设为 300）
PATIENCE = 30  # 早停耐心值：多少轮没进步就停
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

os.makedirs('models', exist_ok=True)  # 确保有保存权重的文件夹


# ============================================================================
#基线模型全家桶 (结构定义)
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


# ============================================================================
# 数据流引擎
# ============================================================================
def load_and_prepare_data():
    print(" [1/3] 正在加载硬盘上的真实 EEG 数据...")
    data_loader = EEGDataLoader(CONFIG)
    raw_samples = data_loader.load_eeg_data()
    data_processor = DataProcessor(CONFIG)
    processed_samples = data_processor.preprocess_data(raw_samples)
    processed_samples = data_processor.create_epochs(processed_samples)
    full_data, full_sessions, full_labels = data_processor.getdate(processed_samples)

    print("⚙ [2/3] 执行 BR-TAD 净化与滤波...")
    nyq = 0.5 * 500
    b, a = butter(4, [8.0 / nyq, 35.0 / nyq], btype='band')
    br_tad_cfg = CONFIG.get('br_tad_config', {})
    all_ch_names = processed_samples[0]['epochs_processed'].ch_names
    center_channels = br_tad_cfg['center_channels']
    center_indices = [all_ch_names.index(ch) for ch in center_channels]
    br_tad_cfg['all_channels'] = all_ch_names

    engine = Orthogonal_Source_BR_TAD_Engine(br_tad_cfg)
    _, X_clean, _ = engine.process_all(full_data)
    X_filtered = filtfilt(b, a, X_clean, axis=-1).astype(np.float32)
    X_10ch = X_filtered[:, center_indices, :]

    print("[3/3] 划分训练集、验证集、测试集...")
    return prepare_dl_data(X_10ch, full_sessions, full_labels, copy.deepcopy(CONFIG["data_selection"]))


# ============================================================================
# 训练、测试、绘图引擎
# ============================================================================
def train_and_eval_model(model_class, name, data_splits, device):
    X_tr, X_va, X_te, y_tr, y_va, y_te = data_splits
    n_ch, n_times = X_tr.shape[1], X_tr.shape[2]
    n_cls = len(np.unique(y_tr))

    print(f"\n=============================================")
    print(f"🚀 开始集训: {name}")
    print(f"=============================================")

    # 初始化模型
    if name == "Standard EEGNet":
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

    # 优化器与损失函数
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

        # 验证
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

        if epoch % 10 == 0:
            print(
                f"  [Epoch {epoch:03d}] Val Acc: {val_acc * 100:.2f}% | Best: {best_acc * 100:.2f}% | EarlyStop: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(f"  ⏹️ 触发早停 (Patience {PATIENCE})")
            break

    # 保存权重
    weight_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    weight_path = f"models/{weight_name}.pth"
    if name == "MS-EEGNet (Proposed)": weight_path = "models/eegnet_best_10ch.pth"
    torch.save(best_weights, weight_path)

    # ---- 测试真实准确率 ----
    model.load_state_dict(best_weights)
    model.eval()
    X_te_t = torch.from_numpy(X_te[:, np.newaxis, :, :]).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)
    with torch.no_grad():
        test_preds = model(X_te_t).argmax(dim=1)
        final_test_acc = (test_preds == y_te_t).sum().item() / len(y_te) * 100.0
    print(f"   最终独立盲测准确率: {final_test_acc:.2f}%")

    # ---- 测速 ----
    dummy_input = torch.randn(1, 1, n_ch, n_times).to(device)
    for _ in range(20): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()

    latency = (time.perf_counter() - start_time) * 1000 / 100
    print(f"  ⏱️ 单次推理延迟: {latency:.2f} ms")

    return params, final_test_acc, latency


def plot_pure_pareto(results):
    print("\n 正在绘制 Pareto 大图...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    accuracies = [r["accuracy"] for r in results]
    latencies = [r["latency"] for r in results]
    params = [r["params"] for r in results]
    colors = [r["color"] for r in results]

    # 调整气泡大小缩放系数，确保 28万参数的 MS-EEGNet 不会由于太大而遮挡文字
    sizes = [np.sqrt(p) * 1.0 for p in params]

    # 绘制气泡
    ax.scatter(latencies, accuracies, s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=1.2)

    # --- 核心改进：精准标签定位 ---
    for i, r in enumerate(results):
        name = r["name"]
        x, y = latencies[i], accuracies[i]

        if "Proposed" in name:

            ax.scatter(x, y, s=sizes[i] * 1.5, c='none', edgecolors='red', linewidth=2, linestyle='--')
            ax.annotate(name, (x, y),
                        xytext=(0, 15),  # 在气泡正上方 15pt 处
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='#D32F2F')
        else:

            ax.annotate(name, (x, y),
                        xytext=(5, -5),  # 向右下偏移 5pt
                        textcoords='offset points',
                        ha='left', va='top',
                        fontsize=9, alpha=0.9)

    ax.set_xlabel('Single-Trial Inference Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=13, fontweight='bold')

    # 自动调整坐标轴范围
    ax.set_xlim(min(latencies) * 0.5, max(latencies) * 1.1)
    ax.set_ylim(min(accuracies) * 0.98, 100.5)

    plt.tight_layout()
    plt.savefig("./Pareto_Frontier_Refined.png")
    print("✅ 标签对齐版大图已保存至: ./Pareto_Frontier_Refined.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用 {device} 进行全自动训练流水线\n")

    # 1. 准备数据
    data_splits = load_and_prepare_data()

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
        "MS-EEGNet (Proposed)": MS_EEGNet
    }

    COLOR_MAP = {
        "ShallowConvNet": '#90CAF9', "DeepConvNet": '#FFA726', "Standard EEGNet": '#29B6F6',
        "ATCNet": '#AB47BC', "TCNet_Fusion": '#26A69A', "EEGTCNet": '#5C6BC0',
        "MBEEG_SENet": '#8D6E63', "EEGNeX": '#78909C', "MS-EEGNet (Proposed)": '#D32F2F'
    }

    # 3. 遍历训练
    results = []
    for name, cls in MODELS.items():
        params, acc, lat = train_and_eval_model(cls, name, data_splits, device)
        results.append({
            "name": name,
            "latency": lat,
            "params": params,
            "accuracy": acc,
            "color": COLOR_MAP[name]
        })

    # 4. 绘图
    plot_pure_pareto(results)