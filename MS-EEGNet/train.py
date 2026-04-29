import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.signal import decimate
from scipy.linalg import inv, sqrtm
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
from br_tad_engine import Orthogonal_Source_BR_TAD_Engine

# --- 依赖引入 ---
try:
    from EEGNet_53 import EEGNet
    from data_loader import EEGDataLoader
    from data_processor import DataProcessor
    from config import CONFIG
except ImportError as e:
    print(f" 缺少必要文件: {e}")
    exit()

# 设置环境
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 绘图风格设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 适配中文
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"随机种子已设置为: {seed}")


# ==========================================
# 1. 核心数学工具
# ==========================================
def apply_trial_ea_single(trial, n_channels=16):
    cov = np.dot(trial, trial.T) / trial.shape[1]
    try:
        r_inv_sqrt = inv(sqrtm(cov + 1e-9 * np.eye(n_channels))).real
        return np.dot(r_inv_sqrt, trial)
    except Exception:
        return trial


def apply_channel_wise_scaling_single(trial):
    ch_means = trial.mean(axis=-1, keepdims=True)
    ch_stds = trial.std(axis=-1, keepdims=True) + 1e-8
    return (trial - ch_means) / ch_stds


def apply_realtime_style_preprocess(data_3d, use_ea=True):
    if data_3d is None or len(data_3d) == 0: return data_3d
    output_data = np.zeros_like(data_3d)
    for i in range(len(data_3d)):
        trial = data_3d[i]
        if use_ea: trial = apply_trial_ea_single(trial)
        trial = apply_channel_wise_scaling_single(trial)
        output_data[i] = trial
    return output_data


def apply_max_norm_constraints(model, max_norm=1.0):
    for name, param in model.named_parameters():
        if 'block2' in name or 'spatial_conv' in name:
            if param.ndim >= 2:
                with torch.no_grad():
                    param.data.copy_(torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm))


# ==========================================
# 2. 绘图工具 (已修改：支持 Acc 和 Loss 双对比)
# ==========================================
def save_training_curves(history, save_dir):
    """绘制并保存训练 Loss 和 Accuracy 对比曲线"""
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 6))

    # --- 子图 1: Loss 对比 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')

    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 子图 2: Accuracy 对比 ---
    plt.subplot(1, 2, 2)

    if 'train_acc' in history:
        plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"训练曲线已保存至: {save_path}")


def save_confusion_matrix(model, loader, device, class_names, save_dir):
    """绘制并保存混淆矩阵"""
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = by.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"混淆矩阵已保存至: {save_dir}/confusion_matrix.png")


# ==========================================
# 3. 数据准备逻辑
# ==========================================
def create_windows(data, labels, w_size, w_step):
    """
    辅助函数：对数据进行滑动窗口切片
    data shape: (N_trials, Channels, Time)
    """
    n_trials = data.shape[0]
    n_times = data.shape[2]

    X_list = []
    y_list = []

    # 遍历每一个 Trial
    for i in range(n_trials):
        start = 0
        # 滑动窗口
        while start + w_size <= n_times:
            # 截取 [start : start+w_size]
            slice_data = data[i, :, start:start + w_size]
            X_list.append(slice_data)
            y_list.append(labels[i])
            start += w_step

    return np.array(X_list), np.array(y_list)


def prepare_dl_data(data, session_ids, labels, data_config):
    print("\n[Data] 准备深度学习数据 (严格 Train/Val/Test 三集划分)...")
    import copy
    data_config = copy.deepcopy(data_config)
    labels, session_ids, data = np.array(labels), np.array(session_ids), np.array(data)

    # 1. 降采样
    ds_cfg = data_config.get("resample_config", {})
    if ds_cfg.get("enabled", False):
        q = int(ds_cfg.get("original_fs", 500) / ds_cfg.get("target_fs", 125))
        data = decimate(data, q=q, axis=-1)
        if "window_config" in data_config:
            data_config["window_config"]["window_size"] //= q
            data_config["window_config"]["step"] //= q

    # 2. 严格的三集划分 (Train / Val / Test)
    if data_config.get("split_mode") == "session" and len(np.unique(session_ids)) >= 3:
        # 切出 15% 的独立 Test 集
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_val_idx, test_idx = next(gss1.split(labels, groups=session_ids))

        # 在剩下的数据中，切出 20% 作为 Val 集
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx_rel, val_idx_rel = next(gss2.split(labels[train_val_idx], groups=session_ids[train_val_idx]))

        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        print("\n" + "-" * 50)
        print(" 跨场次切分名单 (Seed=42):")
        print(f"Train Sessions ({len(np.unique(session_ids[train_idx]))}): {np.unique(session_ids[train_idx])}")
        print(f"Val Sessions   ({len(np.unique(session_ids[val_idx]))}): {np.unique(session_ids[val_idx])}")
        print(f"Test Sessions  ({len(np.unique(session_ids[test_idx]))}): {np.unique(session_ids[test_idx])}")
        print("-" * 50 + "\n")
    else:
        print("Session数量不足，回退到严格的随机分层三集切分...")
        train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.15, stratify=labels,
                                                   random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, stratify=labels[train_val_idx],
                                              random_state=42)

    # 3. 数据平衡 (RUS)
    rus = RandomUnderSampler(random_state=42)
    idx_tr_res, y_train = rus.fit_resample(train_idx.reshape(-1, 1), labels[train_idx])
    idx_va_res, y_val = rus.fit_resample(val_idx.reshape(-1, 1), labels[val_idx])
    idx_te_res, y_test = rus.fit_resample(test_idx.reshape(-1, 1), labels[test_idx])

    X_train_base, X_val_base, X_test_base = data[idx_tr_res.flatten()], data[idx_va_res.flatten()], data[
        idx_te_res.flatten()]

    # 4. 开窗切片
    win_cfg = data_config.get("window_config", {})
    if win_cfg.get("enabled", False):
        w_size, w_step = win_cfg.get("window_size", 1000), win_cfg.get("step", 200)
        n_times = X_train_base.shape[2]

        def apply_windowing(X_base, y_base):
            X_list, y_list = [], []
            for i in range(len(X_base)):
                start = 0
                while start + w_size <= n_times:
                    X_list.append(X_base[i, :, start:start + w_size])
                    y_list.append(y_base[i])
                    start += w_step
            return np.array(X_list), np.array(y_list)

        X_train, y_train = apply_windowing(X_train_base, y_train)
        c_start = (n_times - w_size) // 2
        X_val, X_test = X_val_base[:, :, c_start:c_start + w_size], X_test_base[:, :, c_start:c_start + w_size]
    else:
        X_train, X_val, X_test = X_train_base, X_val_base, X_test_base

    # 5. 标准化
    use_ea = data_config.get("use_ea", True)
    X_train = apply_realtime_style_preprocess(X_train, use_ea=use_ea)
    X_val = apply_realtime_style_preprocess(X_val, use_ea=use_ea)
    X_test = apply_realtime_style_preprocess(X_test, use_ea=use_ea)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# 4. 训练核心
# ==========================================
def train_dl_model(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, class_names, max_epochs=300, patience=20, save_name="model.pth", results_dir="results"):
    print("\n[Start] 启动深度学习训练流程")
    print(f" 配置: Max Epochs={max_epochs} | Patience={patience} | 保存至: {save_name}")

    X_train_t = torch.from_numpy(X_train[:, np.newaxis, :, :]).float()
    X_val_t = torch.from_numpy(X_val[:, np.newaxis, :, :]).float().to(device)
    X_test_t = torch.from_numpy(X_test[:, np.newaxis, :, :]).float().to(device)
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    model = EEGNet(n_channels=X_train.shape[1], n_classes=num_classes, n_times=X_train.shape[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    # 记录4个核心指标
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n[Phase 1] 训练...")
    # 打印表头
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Val Loss':<8} | {'Val Acc':<8} | {'Status'}")
    print("-" * 80)

    for epoch in range(1, max_epochs + 1):
        # === 训练阶段 ===
        model.train()
        train_loss_sum = 0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # # Mixup
            # lam = np.random.beta(0.4, 0.4)
            # idx = torch.randperm(bx.size(0)).to(device)
            # mixed_x = lam * bx + (1 - lam) * bx[idx]
            # y_a, y_b = by, by[idx]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 放宽最大范数限制，给全连接层更多表达空间
            apply_max_norm_constraints(model, max_norm=2.0)

            train_loss_sum += loss.item()

            # ★ 此时的训练准确率，就是模型真实的拟合能力了！
            predicted = out.argmax(dim=1)
            train_correct += (predicted == by).sum().item()
            train_total += by.size(0)

        scheduler.step()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_acc = train_correct / train_total

        # === 验证阶段 ===
        model.eval()
        val_loss_sum = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                outputs = model(vx)

                # ★ 计算验证Loss
                v_loss = criterion(outputs, vy)
                val_loss_sum += v_loss.item()

                v_predicted = outputs.argmax(dim=1)
                val_correct += (v_predicted == vy).sum().item()
                val_total += vy.size(0)

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_acc = val_correct / val_total

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # 检查最佳模型
        status = ""
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            status = "⭐ Best"
        else:
            early_stop_counter += 1
            status = f"Wait {early_stop_counter}"

        # 格式化打印
        print(
            f"{epoch:<6} | {avg_train_loss:.4f}   | {avg_train_acc:.4f}   | {avg_val_loss:.4f}   | {avg_val_acc:.4f}   | {status}")

        if early_stop_counter >= patience:
            print(f"\n⏹️ 早停触发 (Early Stopping): 验证集准确率在 {patience} 个 Epoch 内未提升")
            break

    # 保存最佳模型
    model.load_state_dict(best_weights)
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), f"models/{save_name}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(dim=1)
        final_test_acc = (test_preds == y_test_t).sum().item() / len(y_test_t)

    print(f"\n独立测试集 (Test Set) 盲测最终准确率: {final_test_acc * 100:.2f}%")

    save_training_curves(history, results_dir)
    save_confusion_matrix(model, val_loader, device, class_names, results_dir)

    return final_test_acc


if __name__ == "__main__":
    setup_seed(42)

    # 1. 加载配置参数
    dl_config = CONFIG.get('deep_learning', {})
    max_epochs = dl_config.get('num_epochs', 300)
    patience = dl_config.get('patience', 20)

    print("正在加载原始 EEG 数据...")
    data_loader = EEGDataLoader(CONFIG)
    raw_samples = data_loader.load_eeg_data()

    print("执行基础预处理...")
    data_processor = DataProcessor(CONFIG)
    processed_samples = data_processor.preprocess_data(raw_samples)
    processed_samples = data_processor.create_epochs(processed_samples)

    full_data, full_sessions, full_labels = data_processor.getdate(processed_samples)

    label_map = CONFIG['label_settings']['fixed_mapping']
    idx_to_label = {v: k.replace('ImageStart_', '') for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(class_names)

    # 准备目标带通滤波器
    from scipy.signal import butter, filtfilt

    target_fs = 500
    nyq = 0.5 * target_fs
    b, a = butter(4, [8.0 / nyq, 35.0 / nyq], btype='band')

    # =================================================================
    # 实验 A：Baseline - 无 BR-TAD (16 通道完整特征)
    # =================================================================
    print("\n" + "=" * 65)
    print("[实验 A] 启动 Baseline 训练：无 BR-TAD (16 通道含边缘噪声)")
    print("=" * 65)

    # 1. 直接带通滤波
    X_filtered_A = filtfilt(b, a, full_data, axis=-1).astype(np.float32)

    # 2. 划分为训练/验证/测试集
    X_tr_A, X_va_A, X_te_A, y_tr_A, y_va_A, y_te_A = prepare_dl_data(X_filtered_A, full_sessions, full_labels,
                                                                     CONFIG["data_selection"])

    # 3. 完整补齐参数的调用！
    acc_A = train_dl_model(
        X_tr_A, X_va_A, X_te_A, y_tr_A, y_va_A, y_te_A,
        num_classes, class_names,
        max_epochs, patience,
        save_name="eegnet_best_16ch.pth",
        results_dir="results/Baseline_16ch"
    )

    # =================================================================
    # 实验 B：Proposed - 有 BR-TAD (10 通道核心皮层)
    # =================================================================
    print("\n" + "=" * 65)
    print("🚀 [实验 B] 启动 Proposed 训练：BR-TAD 洗涤 + 空间截断 (10 通道)")
    print("=" * 65)

    br_tad_cfg = CONFIG.get('br_tad_config', {})
    mne_ch_names = processed_samples[0]['epochs_processed'].ch_names
    br_tad_cfg['all_channels'] = mne_ch_names

    # 1. BR-TAD 物理洗涤
    engine = Orthogonal_Source_BR_TAD_Engine(br_tad_cfg)
    _, X_clean, _ = engine.process_all(full_data)

    # 2. 后置滤波
    X_clean_filtered = filtfilt(b, a, X_clean, axis=-1).astype(np.float32)

    # 3. 空间截断 (仅保留核心 10 个通道)
    center_channels = br_tad_cfg['center_channels']
    center_indices = [mne_ch_names.index(ch) for ch in center_channels]
    X_filtered_B = X_clean_filtered[:, center_indices, :]
    print(f"🔪 空间截断完成：输入通道数变为 {X_filtered_B.shape[1]}")

    # 4. 划分为训练/验证/测试集
    X_tr_B, X_va_B, X_te_B, y_tr_B, y_va_B, y_te_B = prepare_dl_data(X_filtered_B, full_sessions, full_labels,
                                                                     CONFIG["data_selection"])

    # 5. 完整补齐参数的调用！
    acc_B = train_dl_model(
        X_tr_B, X_va_B, X_te_B, y_tr_B, y_va_B, y_te_B,
        num_classes, class_names,
        max_epochs, patience,
        save_name="eegnet_best_10ch.pth",
        results_dir="results/BRTAD_10ch"
    )

    # =================================================================
    # 终极消融实验结果汇总
    # =================================================================
    print("\n" + "🌟" * 25)
    print(" 端到端消融实验结果总汇 (独立盲测) ")
    print("🌟" * 25)
    print(f" Baseline (16通道, 含边缘噪声) 独立盲测准确率: {acc_A * 100:.2f}%")
    print(f"Proposed (10通道, 纯净脑皮层) 独立盲测准确率: {acc_B * 100:.2f}%")
    print("\n模型和结果已分别保存在 'models' 和 'results' 文件夹下。")
    print("您现在可以直接去运行 `plot_saliency_topomap.py` 生成对比图了！")