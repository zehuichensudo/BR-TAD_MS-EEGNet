import numpy as np
import matplotlib.pyplot as plt
import mne
import tensorflow as tf
import os
from models import ATCNet_
from preprocess import get_data


def compute_saliency_map(model, X_class_data, y_true_labels, class_idx):
    """Grad * Input 归因法"""
    predictions = model.predict(X_class_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    correct_mask = (predicted_classes == class_idx)
    correct_idx = np.where(correct_mask)[0]

    if len(correct_idx) == 0:
        return np.zeros(X_class_data.shape[2])

    threshold = np.median(confidences[correct_idx])
    final_idx = correct_idx[confidences[correct_idx] >= threshold]
    X_correct = X_class_data[final_idx]

    X_tensor = tf.convert_to_tensor(X_correct, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        class_output = preds[:, class_idx]

    grads = tape.gradient(class_output, X_tensor)
    saliency = tf.abs(grads * X_tensor)
    spatial_importance = np.mean(saliency.numpy(), axis=(0, 1, 3))

    vmax = np.percentile(spatial_importance, 95)
    spatial_importance = np.clip(spatial_importance, 0, vmax)
    spatial_importance = (spatial_importance - np.min(spatial_importance)) / \
                         (np.max(spatial_importance) - np.min(spatial_importance) + 1e-9)

    spatial_importance = spatial_importance ** 2
    return spatial_importance


def load_model_and_data(subject_id, path_weights, params):
    n_ch = 14 if params['use_14'] else 22
    print(f"⏳ Loading {params['name']} (Channels={n_ch})...")

    _, _, _, X_test, y_test, _ = get_data(
        path='./data/', subject=subject_id, dataset='BCI2a', isStandard=True,
        use_14_channels=params['use_14'],
        apply_bandpass=params['bandpass'],
        use_br_tad=params['br_tad'],
        add_noise=params['noise'],
        snr_range=(-34.0, -29.5)
    )

    model = ATCNet_(n_classes=4, in_chans=n_ch, in_samples=1125,
                    n_windows=5, attention='mha', eegn_F1=16, eegn_D=2,
                    eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                    tcn_depth=2, tcn_kernelSize=4, tcn_filters=32,
                    tcn_dropout=0.3, tcn_activation='elu')

    model.load_weights(path_weights)
    return model, X_test, y_test


def plot_paper_figure(figure_name, conditions, title, sub_id=8):
    # 22通道绝对物理坐标系
    ch_names_22 = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                   'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    info_22 = mne.create_info(ch_names=ch_names_22, sfreq=250, ch_types='eeg')
    info_22.set_montage(mne.channels.make_standard_montage('standard_1020'))

    ch_names_14 = ['FC1', 'FCz', 'FC2', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP1', 'CPz', 'CP2', 'Pz']
    idx_14_in_22 = [ch_names_22.index(ch) for ch in ch_names_14]

    classes_labels = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)

    im_for_colorbar = None

    for row_idx, cond in enumerate(conditions):
        weight_path = f"./{cond['path']}/saved models/run-1/subject-{sub_id + 1}.h5"
        try:
            model, X_test, y_test = load_model_and_data(sub_id, weight_path, cond)
            y_true = y_test.argmax(axis=-1) if y_test.ndim > 1 else y_test

            for col_idx, label in enumerate(classes_labels):
                ax = axes[row_idx, col_idx]
                idx = np.where(y_true == col_idx)[0]
                if len(idx) == 0: continue

                sal_map = compute_saliency_map(model, X_test[idx], y_true[idx], col_idx)

                final_map = np.zeros(22)
                if cond['use_14']:
                    final_map[idx_14_in_22] = sal_map
                else:
                    final_map = sal_map

                final_map[final_map < 0.15] = 0.0

                im, _ = mne.viz.plot_topomap(final_map, info_22, axes=ax, cmap='Reds', show=False,
                                             contours=6, extrapolate='head', sphere=(0.0, 0.0, 0.0, 0.095))
                if im_for_colorbar is None: im_for_colorbar = im

                if row_idx == 0: ax.set_title(label, fontsize=16, fontweight='bold', pad=15)
                if col_idx == 0: ax.set_ylabel(cond["name"], fontsize=12, fontweight='bold', labelpad=20)

        except Exception as e:
            print(f"❌ {cond['name']} 失败: {e}")

    if im_for_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im_for_colorbar, cax=cbar_ax)
        cbar.set_label('Saliency Map (Feature Importance)', fontsize=12)

    plt.savefig(f"{figure_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 已生成：{figure_name}.png")


if __name__ == "__main__":
    # ==========================================
    # 图 1: 空间拓扑消融 (EXP 1 vs EXP 3)
    # ==========================================
    fig1_conds = [
        {"name": "EXP 1: 22ch ", "path": "results_1_22ch_Wide_Raw", "use_14": False,
         "bandpass": False, "br_tad": False, "noise": False},
        {"name": "EXP 3: 14ch ", "path": "results_3_14ch_Narrow_Raw", "use_14": True,
         "bandpass": True, "br_tad": False, "noise": False}
    ]
    plot_paper_figure("Figure_1_Spatial_Ablation", fig1_conds,
                      "Fig 1: Spatial Topology Ablation & Physiological Regression")

    # ==========================================
    # 图 2: 窄带实战与救赎 (EXP 7 vs EXP 8)
    # ==========================================
    fig2_conds = [
        {"name": "EXP 7: Noisy Narrow ", "path": "results_7_Noisy_Raw", "use_14": True, "bandpass": True,
         "br_tad": False, "noise": True},
        {"name": "EXP 8: Noisy Narrow + BR-TAD", "path": "results_8_Noisy_BRTAD", "use_14": True, "bandpass": True,
         "br_tad": True, "noise": True}
    ]
    plot_paper_figure("Figure_2_Narrow_Noise_Rescue", fig2_conds,
                      "Fig 2: Limit Disaster Simulation & BR-TAD Rescue (8-35Hz)")

    # ==========================================
    # 图 3: 宽频全解锁挑战 (EXP 5 vs EXP 6)
    # ==========================================
    fig3_conds = [
        {"name": "EXP 5: Noisy Wide ", "path": "results_5_Noisy_Wide_Raw", "use_14": True, "bandpass": False,
         "br_tad": False, "noise": True},
        {"name": "EXP 6: Noisy Wide + BR-TAD", "path": "results_6_Noisy_Wide_BRTAD", "use_14": True, "bandpass": False,
         "br_tad": True, "noise": True}
    ]
    plot_paper_figure("Figure_3_Wide_Noise_Unleashed", fig3_conds,
                      "Fig 3: Full-Band Unleashed without BP Filter (0.5-100Hz)")