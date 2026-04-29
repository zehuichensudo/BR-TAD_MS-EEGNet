import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import copy

from config import CONFIG
from data_loader import EEGDataLoader
from data_processor import DataProcessor
from train import prepare_dl_data
from EEGNet_53 import EEGNet
from br_tad_engine import Orthogonal_Source_BR_TAD_Engine
from scipy.signal import butter, filtfilt

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def get_saliency_map(model, X_tensor, y_tensor, device):
    """提取显著性图：引入 XAI 指数锐化，压制底噪"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        correct_idx = (preds == y_tensor).nonzero(as_tuple=True)[0]
        if len(correct_idx) == 0: return np.zeros(X_tensor.shape[2])

        correct_probs = probs[correct_idx, y_tensor[correct_idx]]
        k = max(1, int(len(correct_idx) * 0.3))
        topk_idx = correct_idx[torch.topk(correct_probs, k).indices]

    X_best = X_tensor[topk_idx].clone().detach().requires_grad_(True)
    y_best = y_tensor[topk_idx]

    out_best = model(X_best)
    score = out_best.gather(1, y_best.view(-1, 1)).squeeze()
    model.zero_grad()
    score.sum().backward()

    saliency = torch.abs(X_best * X_best.grad)
    spatial_weights = saliency.mean(dim=(0, 1, 3)).cpu().detach().numpy()

    if np.max(spatial_weights) > 0:
        spatial_weights = spatial_weights / np.max(spatial_weights)

    # XAI 指数锐化
    spatial_weights = spatial_weights ** 2.0
    return spatial_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_WITHOUT_BRTAD = "./models/eegnet_best_16ch.pth"
    MODEL_WITH_BRTAD = "./models/eegnet_best_10ch.pth"

    if not os.path.exists(MODEL_WITHOUT_BRTAD) or not os.path.exists(MODEL_WITH_BRTAD):
        print("❌ 错误：找不到模型！")
        exit()

    print("📥 加载基础数据...")
    data_loader = EEGDataLoader(CONFIG)
    raw_samples = data_loader.load_eeg_data()
    data_processor = DataProcessor(CONFIG)
    processed_samples = data_processor.preprocess_data(raw_samples)
    processed_samples = data_processor.create_epochs(processed_samples)
    full_data, full_sessions, full_labels = data_processor.getdate(processed_samples)

    nyq = 0.5 * 500
    b, a = butter(4, [8.0 / nyq, 35.0 / nyq], btype='band')

    # === [数据流 A]: 16通道 ===
    X_filtered_A = filtfilt(b, a, full_data, axis=-1).astype(np.float32)
    _, _, X_test_A, _, _, y_test_A = prepare_dl_data(X_filtered_A, full_sessions, full_labels,
                                                     copy.deepcopy(CONFIG["data_selection"]))

    # === [数据流 B]: 10通道 ===
    br_tad_cfg = CONFIG.get('br_tad_config', {})
    all_ch_names = processed_samples[0]['epochs_processed'].ch_names
    center_channels = br_tad_cfg['center_channels']
    center_indices = [all_ch_names.index(ch) for ch in center_channels]

    br_tad_cfg['all_channels'] = all_ch_names
    engine = Orthogonal_Source_BR_TAD_Engine(br_tad_cfg)
    _, X_clean, _ = engine.process_all(full_data)
    X_filtered_B_full = filtfilt(b, a, X_clean, axis=-1).astype(np.float32)
    X_filtered_B_10ch = X_filtered_B_full[:, center_indices, :]
    _, _, X_test_B, _, _, y_test_B = prepare_dl_data(X_filtered_B_10ch, full_sessions, full_labels,
                                                     copy.deepcopy(CONFIG["data_selection"]))

    num_classes = len(np.unique(y_test_A))
    n_times = X_test_A.shape[2]

    model_A = EEGNet(n_channels=16, n_classes=num_classes, n_times=n_times).to(device)
    model_A.load_state_dict(torch.load(MODEL_WITHOUT_BRTAD, map_location=device))
    model_B = EEGNet(n_channels=10, n_classes=num_classes, n_times=n_times).to(device)
    model_B.load_state_dict(torch.load(MODEL_WITH_BRTAD, map_location=device))


    virtual_fence = ['Fp1', 'Fp2', 'F7', 'F8', 'P7', 'P8', 'Oz']
    viz_ch_names = all_ch_names + virtual_fence  # 变成了 23 通道

    montage = mne.channels.make_standard_montage('standard_1020')
    info_viz = mne.create_info(ch_names=viz_ch_names, sfreq=125, ch_types='eeg')
    info_viz.set_montage(montage)

    label_map = CONFIG['label_settings']['fixed_mapping']
    idx_to_label = {v: k.replace('ImageStart_', '').upper() for k, v in label_map.items()}

    fig, axes = plt.subplots(2, 7, figsize=(26, 8))

    axes[0, 0].set_ylabel("Without BR-TAD\n(16 Channels)", fontsize=16, fontweight='bold', labelpad=20)
    axes[1, 0].set_ylabel("With BR-TAD\n(10 Center Channels)", fontsize=16, fontweight='bold', labelpad=20)

    X_tensor_A = torch.from_numpy(X_test_A[:, np.newaxis, :, :]).float().to(device)
    y_tensor_A = torch.from_numpy(y_test_A).long().to(device)
    X_tensor_B = torch.from_numpy(X_test_B[:, np.newaxis, :, :]).float().to(device)
    y_tensor_B = torch.from_numpy(y_test_B).long().to(device)

    for class_idx in range(7):
        class_name = idx_to_label[class_idx]
        print(f"绘制意图类别: {class_name}...")
        axes[0, class_idx].set_title(class_name, fontsize=18, fontweight='bold', pad=15)

        # ====== Row 0: Without BR-TAD (填充到 23 维防漏网格) ======
        idx_A = (y_tensor_A == class_idx)
        if len(X_tensor_A[idx_A]) > 0:
            weights_A_16 = get_saliency_map(model_A, X_tensor_A[idx_A], y_tensor_A[idx_A], device)

            # 把 16个通道的值塞进去，那 7 个防漏电极自动是 0.0
            weights_A_viz = np.zeros(len(viz_ch_names))
            for i, ch in enumerate(all_ch_names):
                weights_A_viz[viz_ch_names.index(ch)] = weights_A_16[i]

            im, _ = mne.viz.plot_topomap(weights_A_viz, info_viz, axes=axes[0, class_idx], show=False,
                                         cmap='Reds', vlim=(0, 1), contours=4, sphere=0.15,
                                         res=300, extrapolate='head')

            # ====== Row 1: With BR-TAD  ======
        idx_B = (y_tensor_B == class_idx)
        if len(X_tensor_B[idx_B]) > 0:
            weights_B_10 = get_saliency_map(model_B, X_tensor_B[idx_B], y_tensor_B[idx_B], device)

            weights_B_viz = np.zeros(len(viz_ch_names))
            for i, ch in enumerate(center_channels):
                weights_B_viz[viz_ch_names.index(ch)] = weights_B_10[i]

            im, _ = mne.viz.plot_topomap(weights_B_viz, info_viz, axes=axes[1, class_idx], show=False,
                                         cmap='Reds', vlim=(0, 1), contours=4, sphere=0.15,
                                         res=300, extrapolate='head')

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label('Feature Importance (Grad * Input)', fontsize=14)
    plt.subplots_adjust(left=0.08, right=0.91, wspace=0.1, hspace=0.2)

    save_path = "./FIGURES/Saliency/Comparison_Homunculus_Ultimate.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"\n全景圆形靶心图生成完毕！请查看: {save_path}")