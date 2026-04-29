import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


FORCE_CPU_MODE = False
GPU_READY = False
cp = np

if not FORCE_CPU_MODE:
    if sys.platform == 'win32':
        conda_dll_path = os.path.join(sys.prefix, 'Library', 'bin')
        if os.path.exists(conda_dll_path):
            try:
                os.add_dll_directory(conda_dll_path)
            except Exception:
                pass
    try:
        import cupy as cp

        _ = cp.array([1.0], dtype=cp.float32).var()
        GPU_READY = True
        print("⚡ [BR-TAD] GPU 极限引擎已激活！")
    except Exception as e:
        print(f"🐢 [BR-TAD] 未检测到 GPU，已降级为 CPU 矢量化模式。")
        import numpy as cp

        GPU_READY = False
else:
    print("🚀 [强制指令]：当前处于纯 CPU 矢量化模式。")

# ============================================================================
# BCI-IV-2a 专属洗涤配置
# ============================================================================
BCI2A_CONFIG = {
    'fs': 250,  # 2a 的采样率是 250Hz
    'edge_channels': ['Fz', 'FC3', 'FC4', 'CP3', 'CP4', 'P1', 'P2', 'POz'],
    'center_channels': [
        'FC1', 'FCz', 'FC2', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP1', 'CPz', 'CP2', 'Pz'
    ],
    'all_channels': [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
}


# ============================================================================
# 🧠 BR-TAD 终极完全体：动态滑动窗口 + 包络追踪
# ============================================================================
class Orthogonal_Source_BR_TAD_Engine:
    def __init__(self, config):
        self.fs = config['fs']
        self.ch_names = config['all_channels']
        self.edge_idx = [self.ch_names.index(ch) for ch in config['edge_channels']]
        self.center_idx = [self.ch_names.index(ch) for ch in config['center_channels']]

        self.win_len = int(0.5 * self.fs)
        self.step = int(0.1 * self.fs)
        self.hann = np.hanning(self.win_len).astype(np.float32)
        self.eps = np.finfo(np.float32).eps

    def _unmix_core(self, c_sig, e_sigs_g, e_envs_g, c_sig_high, e_sigs_high_g, c_sig_brain, e_sigs_brain_g, c_sig_uh,
                    e_sigs_uh_g):
        n_epochs, n_times = c_sig.shape
        n_edges = e_sigs_g.shape[1]
        pad_len = self.win_len

        def pad_gpu(d):
            d_g = cp.asarray(d)
            pad_cfg = ((0, 0), (pad_len, pad_len)) if d_g.ndim == 2 else ((0, 0), (0, 0), (pad_len, pad_len))
            return cp.pad(d_g, pad_cfg, mode='reflect')


        c_p_m = pad_gpu(c_sig)
        c_p_h = pad_gpu(c_sig_high)
        c_p_b = pad_gpu(c_sig_brain)
        c_p_uh = pad_gpu(c_sig_uh)


        e_p_m = e_sigs_g
        e_p_h = e_sigs_high_g
        e_p_b = e_sigs_brain_g
        e_p_uh = e_sigs_uh_g
        e_env_p = e_envs_g


        hann_d = cp.asarray(self.hann)

        n_times_pad = c_p_h.shape[-1]
        out_pure = cp.zeros_like(c_p_m)
        out_artifacts = cp.zeros_like(e_p_m)  # 🚀 这里已经修复为 3D 形状
        global_high_var = cp.var(c_p_h) + self.eps

        weights_sum = cp.zeros(n_times_pad, dtype=cp.float32)
        for s in range(0, n_times_pad - self.win_len + 1, self.step):
            weights_sum[s:s + self.win_len] += hann_d

        batch_idx = cp.arange(n_epochs)
        epoch_idx_mat = batch_idx[:, None]

        for s in range(0, n_times_pad - self.win_len + 1, self.step):
            e_win = s + self.win_len
            t_idx = cp.arange(s, e_win)[None, :]

            # 提取全频段和各高低频段的当前窗口
            C_m = c_p_m[:, s:e_win].copy()
            C_h, C_b = c_p_h[:, s:e_win].copy(), c_p_b[:, s:e_win].copy()
            C_uh = c_p_uh[:, s:e_win].copy()

            E_m = e_p_m[:, :, s:e_win].copy()
            E_h, E_b = e_p_h[:, :, s:e_win].copy(), e_p_b[:, :, s:e_win].copy()
            E_uh = e_p_uh[:, :, s:e_win].copy()
            E_env = e_env_p[:, :, s:e_win].copy()

            E_env_d = E_env - cp.mean(E_env, axis=-1, keepdims=True)
            active_mask = cp.ones((n_epochs, n_edges), dtype=bool)
            V_W_SUM = cp.zeros(n_epochs, dtype=cp.float32)

            for k in range(n_edges):
                var_E = cp.sum(E_env_d ** 2, axis=-1)
                var_E[~active_mask] = -1
                best_indices = cp.argmax(var_E, axis=1)

                S_m = E_m[batch_idx, best_indices, :]
                S_h = E_h[batch_idx, best_indices, :]
                S_b = E_b[batch_idx, best_indices, :]
                S_uh = E_uh[batch_idx, best_indices, :]
                S_env_d = E_env_d[batch_idx, best_indices, :]
                active_mask[batch_idx, best_indices] = False

                # 用 51-100Hz 算 w
                C_uh_d = C_uh - cp.mean(C_uh, axis=-1, keepdims=True)
                S_uh_d = S_uh - cp.mean(S_uh, axis=-1, keepdims=True)
                var_s_uh = cp.sum(S_uh_d ** 2, axis=-1) + self.eps
                var_c_uh = cp.sum(C_uh_d ** 2, axis=-1) + self.eps
                cov_cs_uh = cp.sum(C_uh_d * S_uh_d, axis=-1)

                r_vals = cov_cs_uh / cp.sqrt(var_c_uh * var_s_uh)
                w = cp.clip(cov_cs_uh / var_s_uh, -1.0, 1.0)

                # 能量防线完好保留
                w[cp.abs(r_vals) < 0.2] = 0.0
                w[var_s_uh < 1.2 * var_c_uh] = 0.0


                C_m_new = C_m - w[:, None] * S_m
                C_h_new = C_h - w[:, None] * S_h
                C_b_new = C_b - w[:, None] * S_b

                act_art = C_m - C_m_new
                C_m, C_h, C_b = C_m_new, C_h_new, C_b_new

                # 计算 V 系数和容积传导叠加
                C_b_d = C_b - cp.mean(C_b, axis=-1, keepdims=True)
                S_b_d = S_b - cp.mean(S_b, axis=-1, keepdims=True)
                kv = cp.sum(C_b_d * S_b_d, axis=-1) / (cp.sum(C_b_d ** 2, axis=-1) + self.eps)
                snr = global_high_var / (cp.var(S_h, axis=-1) + global_high_var + self.eps)
                v = (kv * snr) / (1 + (kv * snr) * w + self.eps)
                V_W_SUM += v * w

                # 所有正交化逻辑完好保留 (🚀 增加全频段的正交化)
                S_m_d = S_m - cp.mean(S_m, axis=-1, keepdims=True)
                var_sm = cp.sum(S_m_d ** 2, axis=-1) + self.eps
                E_m_d = E_m - cp.mean(E_m, axis=-1, keepdims=True)
                wi_m = cp.sum(E_m_d * S_m_d[:, None, :], axis=-1) / var_sm[:, None]
                E_m -= cp.clip(wi_m, -1.0, 1.0)[:, :, None] * S_m[:, None, :]

                S_h_d = S_h - cp.mean(S_h, axis=-1, keepdims=True)
                var_s = cp.sum(S_h_d ** 2, axis=-1) + self.eps

                E_h_d = E_h - cp.mean(E_h, axis=-1, keepdims=True)
                wi = cp.sum(E_h_d * S_h_d[:, None, :], axis=-1) / var_s[:, None]
                E_h -= cp.clip(wi, -1.0, 1.0)[:, :, None] * S_h[:, None, :]
                E_b -= cp.clip(wi, -1.0, 1.0)[:, :, None] * S_b[:, None, :]

                E_uh_d = E_uh - cp.mean(E_uh, axis=-1, keepdims=True)
                wi_uh = cp.sum(E_uh_d * S_uh_d[:, None, :], axis=-1) / var_s_uh[:, None]
                E_uh -= cp.clip(wi_uh, -1.0, 1.0)[:, :, None] * S_uh[:, None, :]

                we = cp.sum(E_env_d * S_env_d[:, None, :], axis=-1) / (
                            cp.sum(S_env_d ** 2, axis=-1)[:, None] + self.eps)
                E_env_d -= we[:, :, None] * S_env_d[:, None, :]

                out_artifacts[epoch_idx_mat, best_indices[:, None], t_idx] += (act_art * hann_d)

            denom = cp.maximum(1 - V_W_SUM, self.eps)
            out_pure[:, s:e_win] += (C_m / denom[:, None]) * hann_d

        safe_w = cp.maximum(weights_sum, self.eps)
        f_pure = (out_pure / safe_w)[:, pad_len:-pad_len]
        f_arts = (out_artifacts / safe_w[None, None, :])[:, :, pad_len:-pad_len]

        if GPU_READY:
            return cp.asnumpy(f_pure), cp.asnumpy(f_arts)
        else:
            return f_pure, f_arts

    def process_all(self, X):
        squeeze_flag = False
        if len(X.shape) == 4:
            X_task = np.squeeze(X, axis=1)
            squeeze_flag = True
        else:
            X_task = X

        X_notched = X_task.astype(np.float32)

        def bp(d, l, h):
            nyq = 0.5 * self.fs
            h = min(h, nyq - 1)
            b, a = butter(4, [l / nyq, h / nyq], btype='band')
            return filtfilt(b, a, d, axis=-1).astype(np.float32)

        X_main = X_notched

        # X_low = bp(X_notched, 4, 8)
        X_uh = bp(X_notched, 51, 100)
        X_br = bp(X_notched, 8, 30)
        X_high = bp(X_notched, 8, 100)

        E_env = np.zeros((X_main.shape[0], len(self.edge_idx), X_main.shape[2]), dtype=np.float32)
        for i, e_idx in enumerate(self.edge_idx):
            rect = np.abs(X_uh[:, e_idx, :])
            b_l, a_l = butter(4, 10 / (0.5 * self.fs), btype='low')
            E_env[:, i, :] = filtfilt(b_l, a_l, rect, axis=-1)

        def pre_pad_gpu(d):
            d_g = cp.asarray(d)
            pad_cfg = ((0, 0), (0, 0), (self.win_len, self.win_len))
            return cp.pad(d_g, pad_cfg, mode='reflect')

        e_main_g = pre_pad_gpu(X_main[:, self.edge_idx, :])
        e_high_g = pre_pad_gpu(X_high[:, self.edge_idx, :])
        e_brain_g = pre_pad_gpu(X_br[:, self.edge_idx, :])
        e_uh_g = pre_pad_gpu(X_uh[:, self.edge_idx, :])
        e_env_g = pre_pad_gpu(E_env)

        X_eeg_clean = np.copy(X_main)
        X_artifact_full = np.zeros_like(X_main)

        for j, c_idx in enumerate(self.center_idx):
            clean, art = self._unmix_core(
                X_main[:, c_idx, :], e_main_g, e_env_g,
                X_high[:, c_idx, :], e_high_g,
                X_br[:, c_idx, :], e_brain_g,
                X_uh[:, c_idx, :], e_uh_g  # 🚀 移除了 X_low 传参
            )
            X_eeg_clean[:, c_idx, :] = clean
            X_artifact_full[:, c_idx, :] = np.sum(art, axis=1)

        X_final_clean = X_eeg_clean

        if squeeze_flag:
            X_main = np.expand_dims(X_main, axis=1)
            X_final_clean = np.expand_dims(X_final_clean, axis=1)
            X_artifact_full = np.expand_dims(X_artifact_full, axis=1)


        # # =======================================================
        # import matplotlib.pyplot as plt
        # from scipy.signal import welch
        # import sys  # 引入强制退出模块
        #
        # print("\n正在解剖【真实 BCI2a 数据】的 C3 通道...")
        # c3_idx = self.ch_names.index('C3')
        #
        # # 稳妥提取第 1 个 Trial 的 C3 数据
        # if len(X_main.shape) == 4:
        #     raw_c3 = X_main[0, 0, c3_idx, :]
        #     clean_c3 = X_final_clean[0, 0, c3_idx, :]
        #     art_c3 = X_artifact_full[0, 0, c3_idx, :]
        # else:
        #     raw_c3 = X_main[0, c3_idx, :]
        #     clean_c3 = X_final_clean[0, c3_idx, :]
        #     art_c3 = X_artifact_full[0, c3_idx, :]
        #
        # time_axis = np.linspace(0, len(raw_c3) / self.fs, len(raw_c3))
        #
        # plt.figure(figsize=(15, 10))
        #
        # # --- 图 1：时域波形 ---
        # plt.subplot(2, 1, 1)
        # plt.plot(time_axis, raw_c3, label='Raw C3 (Real Data)', alpha=0.7, color='blue')
        # plt.plot(time_axis, clean_c3, label='Cleaned C3 (BR-TAD)', alpha=0.9, color='green', linewidth=2)
        # plt.plot(time_axis, art_c3, label='What BR-TAD Removed', alpha=0.6, color='red', linestyle='--')
        # plt.title("Time Domain Analysis on REAL DATA", fontsize=14)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.legend(loc='upper right')
        # plt.grid(True)
        #
        # # --- 图 2：频域照妖镜 ---
        # plt.subplot(2, 1, 2)
        # f_raw, Pxx_raw = welch(raw_c3, self.fs, nperseg=self.fs)
        # f_clean, Pxx_clean = welch(clean_c3, self.fs, nperseg=self.fs)
        # f_art, Pxx_art = welch(art_c3, self.fs, nperseg=self.fs)
        #
        # plt.semilogy(f_raw, Pxx_raw, label='Raw C3', color='blue')
        # plt.semilogy(f_clean, Pxx_clean, label='Cleaned C3', color='green', linewidth=2)
        # plt.semilogy(f_art, Pxx_art, label='Removed Components', color='red', linestyle='--')
        #
        # plt.axvspan(8, 12, color='orange', alpha=0.2, label='Mu Rhythm (8-12 Hz)')
        # plt.axvspan(13, 30, color='yellow', alpha=0.2, label='Beta Rhythm (13-30 Hz)')
        #
        # plt.xlim(0, 50)
        # plt.title("Frequency Domain: Is the REAL Mu/Beta Rhythm destroyed?", fontsize=14)
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power Spectral Density")
        # plt.legend(loc='upper right')
        # plt.grid(True)
        #
        # plt.tight_layout()
        # plt.savefig('REAL_DATA_Leakage_Analysis.png')
        # print("\n" + "=" * 50)
        # print("✅ 真实数据解剖图谱已生成！请查看根目录的 'REAL_DATA_Leakage_Analysis.png'")
        # print("=" * 50 + "\n")
        #
        # sys.exit(0)  #拍完照当场自尽！
        # =======================================================
        return X_main, X_final_clean, X_artifact_full
