import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from preprocess import get_data
from br_tad import Orthogonal_Source_BR_TAD_Engine, BCI2A_CONFIG


def plot_signal_leakage():
    print("🔍 正在启动数据透视探针...")

    # 1. 抓取 Subject 1 的一小块干净数据进行解剖
    X_train, _, _, _, _, _ = get_data(
        './data/', 0, 'BCI2a', LOSO=False, isStandard=True,
        use_br_tad=False, add_noise=False
    )

    # 取出第一个 Trial 的数据
    X_raw_trial = X_train[0:1]  # 形状: (1, 1, 22, 1125)

    # 2. 挂载洗涤引擎
    engine = Orthogonal_Source_BR_TAD_Engine(BCI2A_CONFIG)

    # 3. 强行洗涤，拦截三件套！
    print("正在执行 BR-TAD 洗涤...")
    X_raw, X_clean, X_art = engine.process_all(X_raw_trial)

    # 提取 C3 通道 (在 all_channels 中，C3 的索引通常是 7)
    # 保险起见，我们根据名字找索引
    c3_idx = BCI2A_CONFIG['all_channels'].index('C3')

    # 压缩掉多余的维度，变成 1D 数组
    raw_c3 = X_raw[0, 0, c3_idx, :] if len(X_raw.shape) == 4 else X_raw[0, c3_idx, :]
    clean_c3 = X_clean[0, 0, c3_idx, :] if len(X_clean.shape) == 4 else X_clean[0, c3_idx, :]
    art_c3 = X_art[0, 0, c3_idx, :] if len(X_art.shape) == 4 else X_art[0, c3_idx, :]

    time_axis = np.linspace(0, len(raw_c3) / 250.0, len(raw_c3))

    # ================= 绘图开始 =================
    plt.figure(figsize=(15, 10))

    # --- 图 1：时域波形对比 ---
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, raw_c3, label='Raw C3 (Pure Brain)', alpha=0.7, color='blue')
    plt.plot(time_axis, clean_c3, label='Cleaned C3 (After BR-TAD)', alpha=0.9, color='green', linewidth=2)
    plt.plot(time_axis, art_c3, label='Subtracted Artifacts (What was removed)', alpha=0.6, color='red', linestyle='--')
    plt.title("Time Domain: Did we subtract the brain signal?", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True)

    # --- 图 2：频域 PSD 对比 (照妖镜) ---
    plt.subplot(2, 1, 2)
    fs = 250
    f_raw, Pxx_raw = welch(raw_c3, fs, nperseg=fs)
    f_clean, Pxx_clean = welch(clean_c3, fs, nperseg=fs)
    f_art, Pxx_art = welch(art_c3, fs, nperseg=fs)

    plt.semilogy(f_raw, Pxx_raw, label='Raw C3', color='blue')
    plt.semilogy(f_clean, Pxx_clean, label='Cleaned C3', color='green', linewidth=2)
    plt.semilogy(f_art, Pxx_art, label='Removed Components', color='red', linestyle='--')

    # 标出最关键的脑电频段！
    plt.axvspan(8, 12, color='orange', alpha=0.2, label='Mu Rhythm (8-12 Hz)')
    plt.axvspan(13, 30, color='yellow', alpha=0.2, label='Beta Rhythm (13-30 Hz)')

    plt.xlim(0, 50)  # 只看 0-50Hz
    plt.title("Frequency Domain (PSD): Is the Mu/Beta Rhythm destroyed?", fontsize=14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('BRTAD_Leakage_Analysis.png')
    print("✅ 分析图谱已生成！请打开文件夹查看 'BRTAD_Leakage_Analysis.png'！")


if __name__ == "__main__":
    plot_signal_leakage()