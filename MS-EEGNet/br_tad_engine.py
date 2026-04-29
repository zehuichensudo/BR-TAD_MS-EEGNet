import os
import glob
import sys
import numpy as np
import pandas as pd
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
        print(" GPU ")
    except Exception as e:
        print(f" CPU ")
        import numpy as cp

        GPU_READY = False
else:
    print("[强制指令]：当前处于纯 CPU 矢量化模式。")

# ============================================================================
# [模块 1] 核心配置文件
# ============================================================================
CONFIG = {
    'fs': 500,
    'data_dir': './DATA',
    'selected_samples': ['sample5'],
    'all_channels': ['F3', 'Fz', 'F4', 'T7', 'C3', 'C1', 'Cz', 'C2', 'C4', 'T8', 'CPz', 'P3', 'Pz', 'P4', 'O2', 'O1'],
    'edge_channels': ['F3', 'F4', 'T7', 'T8', 'O1', 'O2'],
    'center_channels': ['C3', 'C1', 'Cz', 'C2', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'Fz'],
    'tmin': 1.0,
    'tmax': 5.0,
    'task_events': {
        'ImageStart_up': 0, 'ImageStart_down': 1, 'ImageStart_left': 2, 'ImageStart_right': 3,
        'ImageStart_tight': 4, 'ImageStart_release': 5, 'ImageStart_rest': 6
    }
}


# ============================================================================
# [模块 2] BR-TAD 矢量化全正交引擎
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

    def _unmix_core(self, c_sig, e_sigs_g, e_envs_g, c_sig_high, e_sigs_high_g, c_sig_brain, e_sigs_brain_g, c_sig_low):
        n_epochs, n_times = c_sig.shape
        n_edges = e_sigs_g.shape[1]
        pad_len = self.win_len

        def pad_gpu(d):
            d_g = cp.asarray(d)
            pad_cfg = ((0, 0), (pad_len, pad_len)) if d_g.ndim == 2 else ((0, 0), (0, 0), (pad_len, pad_len))
            return cp.pad(d_g, pad_cfg, mode='reflect')

        c_p_h = pad_gpu(c_sig_high)
        c_p_b = pad_gpu(c_sig_brain)
        c_p_low = pad_gpu(c_sig_low)

        e_p_h = e_sigs_high_g
        e_p_b = e_sigs_brain_g
        e_env_p = e_envs_g
        hann_d = cp.asarray(self.hann)

        n_times_pad = c_p_h.shape[-1]
        out_pure = cp.zeros_like(c_p_h)
        out_artifacts = cp.zeros_like(e_p_h)
        global_high_var = cp.var(c_p_h) + self.eps

        weights_sum = cp.zeros(n_times_pad, dtype=cp.float32)
        for s in range(0, n_times_pad - self.win_len + 1, self.step):
            weights_sum[s:s + self.win_len] += hann_d

        batch_idx = cp.arange(n_epochs)
        epoch_idx_mat = batch_idx[:, None]

        for s in range(0, n_times_pad - self.win_len + 1, self.step):
            e_win = s + self.win_len
            t_idx = cp.arange(s, e_win)[None, :]

            C_h, C_b = c_p_h[:, s:e_win].copy(), c_p_b[:, s:e_win].copy()
            E_h, E_b = e_p_h[:, :, s:e_win].copy(), e_p_b[:, :, s:e_win].copy()
            E_env = e_env_p[:, :, s:e_win].copy()

            E_env_d = E_env - cp.mean(E_env, axis=-1, keepdims=True)
            active_mask = cp.ones((n_epochs, n_edges), dtype=bool)
            V_W_SUM = cp.zeros(n_epochs, dtype=cp.float32)

            for k in range(n_edges):
                var_E = cp.sum(E_env_d ** 2, axis=-1)
                var_E[~active_mask] = -1
                best_indices = cp.argmax(var_E, axis=1)

                S_h = E_h[batch_idx, best_indices, :]
                S_b = E_b[batch_idx, best_indices, :]
                S_env_d = E_env_d[batch_idx, best_indices, :]
                active_mask[batch_idx, best_indices] = False

                C_h_d = C_h - cp.mean(C_h, axis=-1, keepdims=True)
                S_h_d = S_h - cp.mean(S_h, axis=-1, keepdims=True)
                var_s = cp.sum(S_h_d ** 2, axis=-1) + self.eps
                cov_cs = cp.sum(C_h_d * S_h_d, axis=-1)

                r_vals = cov_cs / cp.sqrt((cp.sum(C_h_d ** 2, axis=-1) + self.eps) * var_s)
                w = cp.clip(cov_cs / var_s, -1.0, 1.0)
                w[cp.abs(r_vals) < 0.2] = 0.0

                C_h_new = C_h - w[:, None] * S_h
                C_b_new = C_b - w[:, None] * S_b
                C_h_new[(C_h * C_h_new) < 0] = 0.0
                C_b_new[(C_b * C_b_new) < 0] = 0.0

                act_art = C_h - C_h_new
                C_h, C_b = C_h_new, C_b_new

                C_b_d = C_b - cp.mean(C_b, axis=-1, keepdims=True)
                S_b_d = S_b - cp.mean(S_b, axis=-1, keepdims=True)
                kv = cp.sum(C_b_d * S_b_d, axis=-1) / (cp.sum(C_b_d ** 2, axis=-1) + self.eps)
                snr = global_high_var / (cp.var(S_h, axis=-1) + global_high_var + self.eps)
                v = (kv * snr) / (1 + (kv * snr) * w + self.eps)
                V_W_SUM += v * w

                E_h_d = E_h - cp.mean(E_h, axis=-1, keepdims=True)
                wi = cp.sum(E_h_d * S_h_d[:, None, :], axis=-1) / var_s[:, None]
                E_h -= cp.clip(wi, -1.0, 1.0)[:, :, None] * S_h[:, None, :]
                E_b -= cp.clip(wi, -1.0, 1.0)[:, :, None] * S_b[:, None, :]

                we = cp.sum(E_env_d * S_env_d[:, None, :], axis=-1) / (
                            cp.sum(S_env_d ** 2, axis=-1)[:, None] + self.eps)
                E_env_d -= we[:, :, None] * S_env_d[:, None, :]

                out_artifacts[epoch_idx_mat, best_indices[:, None], t_idx] += (act_art * hann_d)

            denom = cp.maximum(1 - V_W_SUM, self.eps)
            out_pure[:, s:e_win] += (C_h / denom[:, None] + c_p_low[:, s:e_win]) * hann_d

        safe_w = cp.maximum(weights_sum, self.eps)
        f_pure = (out_pure / safe_w)[:, pad_len:-pad_len]
        f_arts = (out_artifacts / safe_w[None, None, :])[:, :, pad_len:-pad_len]

        return cp.asnumpy(f_pure), cp.asnumpy(f_arts)

    def process_all(self, X_task):
        mode_str = "GPU 极限引擎" if GPU_READY else " CPU 矢量化"

        b, a_n = iirnotch(50, 20, self.fs)
        X_notched = filtfilt(b, a_n, X_task, axis=-1).astype(np.float32)

        def bp(d, l, h):
            nyq = 0.5 * self.fs
            b, a = butter(4, [l / nyq, h / nyq], btype='band')
            return filtfilt(b, a, d, axis=-1).astype(np.float32)

        X_main = bp(X_notched, 4, 100)
        X_low = bp(X_notched, 4, 8)
        X_high = bp(X_notched, 8, 100)
        X_uh = bp(X_notched, 51, 100)
        X_br = bp(X_notched, 8, 30)

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
        e_env_g = pre_pad_gpu(E_env)

        X_eeg_clean = np.copy(X_main)

        # 拦截 C3 的边缘伪迹
        X_artifact_C3 = np.zeros((X_main.shape[0], len(self.edge_idx), X_main.shape[2]), dtype=np.float32)

        for j, c_idx in enumerate(self.center_idx):
            c_name = self.ch_names[c_idx]
            clean, art = self._unmix_core(X_main[:, c_idx, :], e_main_g, e_env_g,
                                          X_high[:, c_idx, :], e_high_g,
                                          X_br[:, c_idx, :], e_brain_g, X_low[:, c_idx, :])
            X_eeg_clean[:, c_idx, :] = clean

            # 捕获 C3 处理时生成的严格 6 通道伪迹
            if c_name == 'C3':
                X_artifact_C3 = art

        return X_main, bp(X_eeg_clean, 4, 100), X_artifact_C3


# ============================================================================
# [模块 3] 数据提取与主程序
# ============================================================================
class RawDataParser:
    def __init__(self, config):
        self.cfg = config

    def load_and_slice(self):
        print(">>> [步骤1] 提取原始数据...")
        all_x, all_y, s_ids, sess_ids = [], [], [], []
        session_dirs = sorted(glob.glob(os.path.join(self.cfg['data_dir'], 'sample*', 'session*')))
        for sess_dir in session_dirs:
            s_n, sess_n = sess_dir.split(os.sep)[-2:]
            if self.cfg.get('selected_samples') and s_n not in self.cfg['selected_samples']: continue
            eeg_f, evt_f = os.path.join(sess_dir, 'merged_dataCsv.csv'), os.path.join(sess_dir, 'Event.csv')
            if not os.path.exists(eeg_f) or not os.path.exists(evt_f): continue
            ed, ev = pd.read_csv(eeg_f), pd.read_csv(evt_f)
            ts, em = ed['TIMESTAMP'].values, ed[self.cfg['all_channels']].values.T
            spe = int((self.cfg['tmax'] - self.cfg['tmin']) * self.cfg['fs'])
            for _, row in ev.iterrows():
                en = row['Event']
                idx = np.argmin(np.abs(ts - row['TIMESTAMP']))
                si = idx + int(self.cfg['tmin'] * self.cfg['fs'])
                if si + spe >= em.shape[1]: continue
                if en in self.cfg['task_events']:
                    all_x.append(em[:, si:si + spe])
                    all_y.append(self.cfg['task_events'][en])
                    s_ids.append(s_n)
                    sess_ids.append(f"{s_n}_{sess_n}")
        return np.array(all_x, dtype=np.float32), np.array(all_y), np.array(s_ids), np.array(sess_ids)


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("BR-TAD 终极完全体")
    print("=" * 75)

    parser = RawDataParser(CONFIG)
    X_task, y_task, sample_ids, session_ids = parser.load_and_slice()

    if len(X_task) == 0:
        print("❌ 未找到有效数据。")
        exit()

    engine = Orthogonal_Source_BR_TAD_Engine(CONFIG)
    X_raw, X_clean, X_artifact = engine.process_all(X_task)

    save_path = os.path.join(CONFIG['data_dir'], 'ag_ssd_TOTAL_DATA.npz')
    np.savez_compressed(save_path,
                        X_raw=X_raw,
                        X_clean=X_clean,
                        X_artifact=X_artifact,
                        y_task=y_task,
                        sample_ids=sample_ids,
                        session_ids=session_ids,
                        ch_names=CONFIG['all_channels'])

    print(f"\n✅ 全流程处理完成。物理真相包已保存至: {save_path}")