import mne
import numpy as np
import pandas as pd
from scipy import signal

def create_epochs_from_preprocessed_data(data_samples, epoch_params,label_settings):
    """从预处理后的数据创建epochs """
    for i, sample in enumerate(data_samples):
        print(f"  为样本 {i + 1} 创建epochs...")

        if 'raw_processed' in sample and sample['raw_processed'] is not None:
            try:
                raw_processed = sample['raw_processed']
                event_df = sample['event_data']
                eeg_df = sample['eeg_data']
                # 使用预处理后的raw创建epochs
                epochs = create_epochs_from_preprocessed_raw(
                    raw_processed,
                    event_df,
                    eeg_df,
                    tmin=epoch_params['tmin'],
                    tmax=epoch_params['tmax'],
                    baseline=epoch_params['baseline'],
                    label_settings = label_settings
                )
                epochs_crop = epochs.copy()

                epochs_crop.crop(epoch_params['tmin_crop'],epoch_params['tmax_crop'])

                sample.update({
                    "tmin": epoch_params['tmin'],
                    "tmax": epoch_params['tmax'],
                    "tmin_crop": epoch_params['tmin_crop'],
                    "tmax_crop": epoch_params['tmax_crop'],
                    "baseline": epoch_params['baseline']
                })

                if epochs is not None:
                    sample['epochs_processed'] = epochs_crop
                    print(
                        f"    成功创建预处理epochs: {len(epochs)} 个epochs, 时间窗口 {epochs.tmin:.1f}-{epochs.tmax:.1f}秒")

                    # 打印各事件类型的epoch数量
                    for event_name, event_id in epochs.event_id.items():
                        event_epochs = epochs[event_name]
                        print(f"      事件 '{event_name}': {len(event_epochs)} 个epochs")
                else:
                    print("    警告: 无法创建epochs")

            except Exception as e:
                print(f"    创建epochs时出错: {e}")
        else:
            print("    警告: 没有预处理数据可用于创建epochs")


def create_epochs_from_preprocessed_raw(raw_processed, event_df, eeg_df, tmin=0.5, tmax=4.5, baseline=None, label_settings=None):
    """
    从预处理后的Raw对象和事件DataFrame创建Epochs对象
    """
    # 获取预处理数据中的时间戳
    eeg_timestamps = eeg_df.iloc[:, 0].values

    # 创建事件列表
    events = []
    event_type_mapping = label_settings['fixed_mapping']

    used_event_ids = {}

    # 遍历事件数据
    for _, event_row in event_df.iterrows():
        event_type = event_row['Event']

        # 只处理以"ImageStart_"开头的事件
        if event_type.startswith('ImageStart_'):
            # 获取事件时间戳
            event_timestamp = event_row['TIMESTAMP']

            # 在脑电数据中找到最接近的时间戳
            time_diff = np.abs(eeg_timestamps - event_timestamp)
            event_sample = np.argmin(time_diff)

            # 确定事件ID
            if event_type in event_type_mapping:
                event_id_code = event_type_mapping[event_type]
            else:
                # 为未映射的事件类型分配新ID
                if event_type not in used_event_ids:
                    next_id = max(event_type_mapping.values()) + 1 if event_type_mapping else 1
                    event_type_mapping[event_type] = next_id
                    used_event_ids[event_type] = next_id
                event_id_code = event_type_mapping[event_type]

            # 检查事件时间是否在有效范围内（避免在数据开始或结束附近的事件）
            sfreq = raw_processed.info['sfreq']
            start_buffer = abs(tmin) * sfreq if tmin < 0 else 0
            end_buffer = tmax * sfreq

            total_samples = raw_processed.n_times
            if event_sample >= start_buffer and event_sample <= (total_samples - end_buffer):
                events.append([event_sample, 0, event_id_code])
            else:
                print(f"      跳过边界事件: {event_type} at sample {event_sample}")
    # 将 events 列表转换为 MNE Annotations
    sfreq = raw_processed.info['sfreq']
    onsets = [e[0] / sfreq for e in events]
    durations = [5.0] * len(onsets)

    # 建立反向映射，获取标签名称
    rev_mapping = {v: k for k, v in event_type_mapping.items()}
    descriptions = [rev_mapping[e[2]] for e in events]

    # 注入到 raw 对象
    annot = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw_processed.set_annotations(annot)

    print("    已同步将 Annotations 注入 Raw 对象")
    if not events:
        print("警告: 未找到任何有效事件")
        return None

    events = np.array(events).astype(int)
    # 创建事件ID映射
    event_id = {}
    for event_name, event_code in event_type_mapping.items():
        if event_code in events[:, 2]:
            event_id[event_name] = event_code

    # 创建Epochs对象 - 使用预处理后的raw数据
    print(f"    创建epochs: tmin={tmin}, tmax={tmax}, baseline={baseline}")
    try:
        epochs = mne.Epochs(raw_processed, events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True,  # 确保数据被加载到内存
                            event_repeated='drop',  # 或 'merge'
                            # reject=reject,
                            picks='eeg',   # 只选择EEG通道
                            on_missing='warn')  # 处理缺失事件

        # 检查是否有有效的epochs
        if len(epochs) == 0:
            print("    警告: 创建的epochs为空")
            return None

        return epochs

    except Exception as e:
        print(f"    创建Epochs对象失败: {e}")
        return None


def create_evoked_objects(all_samples, config):
    """从epochs创建evoked对象并进行可视化分析 - 修改为整个session平均"""
    print("\n=== 创建Evoked对象 (Session平均) ===")

    evoked_analysis_config = config.get('evoked_analysis', {})
    visualization_config = config.get('visualization', {})

    if not evoked_analysis_config.get('enabled', True):
        print("Evoked分析已禁用")
        return all_samples

    # 按事件类型收集所有epochs
    epochs_by_condition = {}

    for i, sample in enumerate(all_samples):
        if 'epochs_processed' in sample and sample['epochs_processed'] is not None:
            epochs = sample['epochs_processed']
            sample_name = sample['sample_name']

            print(f"处理样本: {sample_name}")

            # 为每个事件类型收集epochs
            for event_name in epochs.event_id.keys():
                try:
                    event_epochs = epochs[event_name]

                    if event_name not in epochs_by_condition:
                        epochs_by_condition[event_name] = []

                    epochs_by_condition[event_name].append(event_epochs)
                    print(f"  添加 {event_name}: {len(event_epochs)} 个epochs")

                except Exception as e:
                    print(f"  处理 {event_name} 失败: {e}")

    if not epochs_by_condition:
        print("警告: 未找到任何有效的epochs数据")
        return all_samples

    # 为每个事件类型创建组合的evoked（整个session平均）
    combined_evoked = {}

    for event_name, epochs_list in epochs_by_condition.items():
        try:
            if len(epochs_list) == 1:
                # 只有一个样本，直接平均
                evoked = epochs_list[0].average()
            else:
                # 合并多个样本的epochs，然后平均
                # 使用mne.concatenate_epochs合并epochs
                from mne import concatenate_epochs
                combined_epochs = concatenate_epochs(epochs_list)
                evoked = combined_epochs.average()

            combined_evoked[event_name] = evoked
            print(f"创建 {event_name} 的evoked: {evoked.data.shape}")

        except Exception as e:
            print(f"创建 {event_name} 的evoked失败: {e}")

    # 可视化组合的evoked
    if combined_evoked and visualization_config.get('plot_evoked', True):
        plot_combined_evoked(combined_evoked, visualization_config, "Session平均")

    # 保存evoked对象
    if evoked_analysis_config.get('save_evoked', True) and combined_evoked:
        save_evoked_objects(combined_evoked, evoked_analysis_config)

    # 将组合evoked存储到第一个样本中，便于后续使用
    if all_samples and combined_evoked:
        all_samples[0]['combined_evoked'] = combined_evoked

    return all_samples


def plot_combined_evoked(combined_evoked, visualization_config, title_suffix=""):
    """绘制组合evoked图像 """
    try:
        import matplotlib.pyplot as plt

        n_conditions = len(combined_evoked)
        if n_conditions == 0:
            return

        # 创建更清晰的图形布局
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'事件相关电位 (ERP) 分析 - {title_suffix}',
                     fontsize=16, fontweight='bold')

        # 1. 所有条件的ERP曲线对比
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        colors = plt.cm.tab10(np.linspace(0, 1, n_conditions))

        for idx, (event_name, evoked) in enumerate(combined_evoked.items()):
            # 使用所有通道的平均值
            mean_data = evoked.data.mean(axis=0)
            std_data = evoked.data.std(axis=0)

            times = evoked.times
            line = ax1.plot(times, mean_data,
                            label=event_name.replace('ImageStart_', ''),
                            color=colors[idx], linewidth=2.5, alpha=0.8)

            # 添加透明度区域显示标准差
            ax1.fill_between(times, mean_data - std_data, mean_data + std_data,
                             alpha=0.2, color=colors[idx])

        ax1.axvline(0, linestyle="--", color="k", linewidth=1, alpha=0.7, label="刺激开始")
        ax1.axhline(0, linestyle="-", color="gray", linewidth=0.5, alpha=0.5)

        ax1.set_xlabel("时间 (秒)", fontsize=12)
        ax1.set_ylabel("振幅 (μV)", fontsize=12)
        ax1.set_title("所有条件的ERP曲线对比（通道平均）", fontsize=14, pad=20)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(evoked.times[0], evoked.times[-1])

        # 2. 选择4个重要条件进行详细分析
        important_conditions = list(combined_evoked.keys())[:4]  # 只取前4个

        # 为每个重要条件绘制单独的时间序列
        for idx, condition in enumerate(important_conditions):
            if idx < 4:  # 确保在4个位置内
                row = (idx // 2) + 1  # 第1-2行
                col = idx % 2  # 第0-1列

                ax = plt.subplot2grid((3, 2), (row, col))
                evoked = combined_evoked[condition]

                # 绘制该条件的所有通道
                for ch_idx in range(min(5, evoked.data.shape[0])):  # 只显示前5个通道
                    ax.plot(evoked.times, evoked.data[ch_idx, :],
                            alpha=0.6, linewidth=1, label=f'Ch{ch_idx + 1}')

                ax.axvline(0, linestyle="--", color="k", alpha=0.7)
                ax.axhline(0, linestyle="-", color="gray", alpha=0.5)
                ax.set_xlabel("时间 (秒)")
                ax.set_ylabel("振幅 (μV)")
                condition_short = condition.replace('ImageStart_', '')
                ax.set_title(f'{condition_short} - 主要通道', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(evoked.times[0], evoked.times[-1])

                # 只在第一个图添加图例
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

        # 3. 拓扑图分析
        plot_evoked_topomaps(combined_evoked, title_suffix)

    except Exception as e:
        print(f"绘制组合evoked图像失败: {e}")
        import traceback
        traceback.print_exc()


def plot_evoked_topomaps(combined_evoked, title_suffix=""):
    """绘制evoked拓扑图 """
    try:
        import matplotlib.pyplot as plt

        # 选择关键时间点
        times = [2.0, 2.5, 3.0, 3.5, 4.0]

        # 为每个条件创建拓扑图
        for condition, evoked in combined_evoked.items():
            try:
                condition_short = condition.replace('ImageStart_', '')

                # 创建图形
                fig, axes = plt.subplots(1, len(times), figsize=(15, 3))
                if len(times) == 1:
                    axes = [axes]

                fig.suptitle(f'{condition_short} - 脑电活动时空分布', fontsize=14, y=1.05)

                # 为每个时间点绘制拓扑图
                for idx, (time_point, ax) in enumerate(zip(times, axes)):
                    try:
                        # 检查时间点是否在范围内
                        if time_point >= evoked.times[0] and time_point <= evoked.times[-1]:
                            # 绘制拓扑图并获取颜色条
                            im, cn = evoked.plot_topomap(
                                times=[time_point],
                                axes=ax,
                                show=False,
                                time_unit='s',
                                size=2,
                                sensors=True,
                                contours=6,
                                time_format=f't = {time_point:.1f}s'
                            )
                            ax.set_title(f'{time_point}s', fontsize=10)
                        else:
                            ax.text(0.5, 0.5, f'时间点\n{time_point}s\n超出范围',
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{time_point}s', fontsize=10)

                    except Exception as e:
                        ax.text(0.5, 0.5, f'绘图错误',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{time_point}s', fontsize=10)

                # 添加颜色条说明
                if 'im' in locals():
                    cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
                    cbar.set_label('振幅 (μV)', rotation=270, labelpad=15)
                    cbar.ax.tick_params(labelsize=8)

                plt.tight_layout()
                plt.show()

                # 打印该条件的统计信息
                print(f"\n{condition_short} 统计:")
                peak_amplitude = np.max(np.abs(evoked.data))
                mean_amplitude = np.mean(np.abs(evoked.data))
                print(f"  峰值振幅: {peak_amplitude:.3f} μV")
                print(f"  平均振幅: {mean_amplitude:.3f} μV")

            except Exception as e:
                print(f"绘制 {condition} 拓扑图失败: {e}")
                continue

        # 4. 条件对比拓扑图
        plot_condition_comparison(combined_evoked, title_suffix)

    except Exception as e:
        print(f"绘制拓扑图失败: {e}")


def plot_condition_comparison(combined_evoked, title_suffix=""):
    """绘制条件对比图 """
    try:
        import matplotlib.pyplot as plt

        # 选择要对比的条件（最多4个）
        conditions = list(combined_evoked.keys())[:4]
        if len(conditions) < 2:
            return

        # 找到共同的时间点
        common_time = 3.0  # 选择中间时间点

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        fig.suptitle(f'条件对比 - 时间点 t={common_time}s', fontsize=16, y=0.95)

        # 绘制每个条件的拓扑图
        for idx, condition in enumerate(conditions):
            if idx < 4:
                evoked = combined_evoked[condition]
                condition_short = condition.replace('ImageStart_', '')

                try:
                    # 绘制拓扑图
                    im, _ = evoked.plot_topomap(
                        times=[common_time],
                        axes=axes[idx],
                        show=False,
                        sensors=True,
                        contours=6
                    )

                    # 添加统计信息
                    data_at_time = evoked.data[:, np.argmin(np.abs(evoked.times - common_time))]
                    max_amp = np.max(np.abs(data_at_time))
                    mean_amp = np.mean(np.abs(data_at_time))

                    axes[idx].set_title(
                        f'{condition_short}\n'
                        f'峰值: {max_amp:.2f}μV\n'
                        f'均值: {mean_amp:.2f}μV',
                        fontsize=11
                    )

                except Exception as e:
                    axes[idx].text(0.5, 0.5, f'无法绘制\n{condition_short}',
                                   ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(condition_short)

        # 添加颜色条说明
        cbar = plt.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label('振幅 (μV)', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)

        # 添加颜色说明
        fig.text(0.02, 0.02,
                 '颜色说明: 红色表示正电位，蓝色表示负电位\n颜色深浅表示电位强度',
                 fontsize=10, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08)
        plt.show()

    except Exception as e:
        print(f"绘制条件对比图失败: {e}")

def save_evoked_objects(combined_evoked, evoked_analysis_config):
    """保存evoked对象到文件 """
    try:
        import os
        import pickle
        from datetime import datetime

        evoked_dir = evoked_analysis_config.get('evoked_dir', 'evoked_results')
        os.makedirs(evoked_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存为pickle文件
        evoked_file = f"{evoked_dir}/session_evoked_{timestamp}.pkl"
        with open(evoked_file, 'wb') as f:
            pickle.dump(combined_evoked, f)

        print(f"Session平均Evoked对象已保存: {evoked_file}")

    except Exception as e:
        print(f"保存evoked对象失败: {e}")

def create_raw_from_dataframe(eeg_df, sfreq=500):
        """从DataFrame创建MNE Raw对象"""
        eeg_data = eeg_df.iloc[:, 2:].values.T
        ch_names = list(eeg_df.columns[2:])
        ch_types = ['eeg'] * len(ch_names)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info)

        return raw

def process_session_files_without_epochs(eeg_file, event_file, sfreq):
        """处理session文件并只创建Raw对象（不创建epochs）"""
        eeg_df = pd.read_csv(eeg_file)
        event_df = pd.read_csv(event_file)

        raw = create_raw_from_dataframe(eeg_df, sfreq=sfreq)

        return raw, event_df, eeg_df


import os
import mne


def preprocess_all_samples(data_samples, preprocess_params, visualize=False, save_dir="processed_raw"):
    """
    预处理所有样本数据，支持断点续传（检查是否存在已处理的文件）
    """
    # 确保存储目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, sample in enumerate(data_samples):
        sample_name = sample['sample_name']
        session_name = sample['session_name']
        save_path = os.path.join(save_dir, f"{sample_name}_{session_name}_raw.fif")

        print(f"\n=== 样本 {i + 1}/{len(data_samples)}: {sample_name} - {session_name} ===")

        raw, event_df, eeg_df = process_session_files_without_epochs(
            sample["eeg_file"], sample["event_file"], 500
        )

        sample.update({
            "raw": raw,
            "event_data": event_df,
            "eeg_data": eeg_df
        })

        # 1. 检测是否存在已处理的文件
        if os.path.exists(save_path):
            print(f"  [跳过] 检测到预处理文件已存在: {save_path}")
            try:
                # 直接读取已存在的预处理文件
                raw_processed = mne.io.read_raw_fif(save_path, preload=True)
                sample['raw_processed'] = raw_processed
                sample['eeg_processed'] = raw_processed.get_data()
                continue  # 进入下一个样本循环
            except Exception as e:
                print(f"  读取旧文件失败，重新处理: {e}")

        try:
            # 执行预处理
            raw_processed = preprocess_raw(raw, preprocess_params)
            # 3. 存储处理后的结果
            # overwrite=True 确保在之前读取失败的情况下可以覆盖
            raw_processed.save(save_path, overwrite=True)
            print(f"  [存储] 预处理完成并保存至: {save_path}")
            sample['raw_processed'] = raw_processed
            sample['eeg_processed'] = raw_processed.get_data()
        except Exception as e:
            print(f"  预处理过程中出错: {e}")
            continue

def preprocess_raw(raw, params):
    raw_working = raw.copy()

    montage = mne.channels.make_standard_montage("standard_1020")
    raw_working.set_montage(montage, on_missing='ignore')


    # 1. 工频滤波 (先处理窄带强噪声)
    if params.get('notch_filter', False):
        raw_working.notch_filter(freqs=params.get('notch_freqs', 50))

    # 2. 带通滤波
    raw_filtered = raw_working.filter(
        l_freq=params['l_freq'],
        h_freq=params['h_freq'],
        fir_design='firwin'
    )

    # 3. ICA 处理
    if params.get('use_ica', False):
        ica = mne.preprocessing.ICA(n_components=params.get('ica_components', 15),
                                    random_state=97, max_iter=800)
        ica.fit(raw_filtered)

        raw_processed = ica.apply(raw_filtered.copy())
    else:
        raw_processed = raw_filtered

    # 4. 重参考
    raw_processed.set_eeg_reference('average', projection=False)  # 直接修改数据

    return raw_processed


def apply_downsampling_to_raw(raw, target_sfreq, method='resample_poly', antialias=True, config=None):
    """
    应用降采样到Raw对象

    参数:
    ----------
    raw : mne.io.Raw
        原始数据
    target_sfreq : float
        目标采样率
    method : str
        降采样方法: 'resample_poly', 'decimate', 'mean_pooling'
    antialias : bool
        是否应用抗混叠滤波
    config : dict
        额外配置参数

    返回:
    ----------
    raw_downsampled : mne.io.Raw
        降采样后的数据
    downsampling_info : dict
        降采样信息字典
    """
    if config is None:
        config = {}

    original_sfreq = raw.info['sfreq']

    # 创建降采样信息字典
    downsampling_info = {
        'original_sfreq': original_sfreq,
        'target_sfreq': target_sfreq,
        'method': method,
        'antialias': antialias,
        'downsample_factor': original_sfreq / target_sfreq if target_sfreq > 0 else 1
    }

    # 计算降采样因子
    downsample_factor = int(original_sfreq / target_sfreq)

    if downsample_factor <= 1:
        print(f"  警告: 降采样因子为{downsample_factor}，跳过降采样")
        downsampling_info['skipped'] = True
        return raw.copy(), downsampling_info

    print(f"  降采样因子: {downsample_factor}")
    downsampling_info['downsample_factor_int'] = downsample_factor

    # 根据选择的方法执行降采样
    if method == 'resample_poly':
        # 使用MNE内置的重采样方法
        print(f"  使用resample_poly方法降采样...")
        try:
            # MNE的resample方法内部会应用抗混叠滤波
            raw_downsampled = raw.copy().resample(
                sfreq=target_sfreq,
                npad='auto'
            )

            # 记录降采样信息
            downsampling_info['applied'] = True
            downsampling_info['success'] = True

            return raw_downsampled, downsampling_info

        except Exception as e:
            print(f"  MNE重采样失败: {e}, 尝试其他方法")
            method = 'decimate'  # 降级到其他方法
            downsampling_info['method'] = 'decimate'  # 更新方法

    if method == 'decimate':
        # 使用scipy的decimate方法
        print(f"  使用decimate方法降采样...")

        # 如果需要抗混叠滤波，decimate会自动应用
        raw_data = raw.get_data()
        n_channels, n_samples = raw_data.shape

        # 计算合适的decimate因子
        if downsample_factor > 13:
            # 多级降采样
            print(f"  使用多级降采样 (因子: {downsample_factor})...")

            # 找到质因数分解
            factors = []
            remaining = downsample_factor
            while remaining > 13:
                # 找到小于等于13的最大因子
                for f in range(13, 1, -1):
                    if remaining % f == 0:
                        factors.append(f)
                        remaining = remaining // f
                        break
                else:
                    # 如果没有找到合适的因子，使用13
                    factors.append(13)
                    remaining = remaining // 13

            if remaining > 1:
                factors.append(remaining)

            print(f"  多级降采样因子: {factors}")
            downsampling_info['multistage_factors'] = factors

            # 逐级降采样
            data_downsampled = raw_data
            current_sfreq = original_sfreq

            for i, factor in enumerate(factors):
                print(f"    第{i + 1}级降采样，因子: {factor}")
                data_downsampled = signal.decimate(
                    data_downsampled,
                    factor,
                    axis=1,
                    ftype='fir'
                )
                current_sfreq = current_sfreq / factor

        else:
            # 单级降采样
            data_downsampled = signal.decimate(
                raw_data,
                downsample_factor,
                axis=1,
                ftype='fir'
            )

        # 创建新的Raw对象
        from mne import create_info
        from mne.io import RawArray

        # 更新通道信息
        ch_names = raw.info['ch_names']
        ch_types = raw.info.get('chs', [{} for _ in ch_names])
        ch_types = [ch.get('kind', 'eeg') for ch in ch_types]

        info_downsampled = create_info(
            ch_names=ch_names,
            sfreq=target_sfreq,
            ch_types=ch_types
        )

        # 复制其他信息
        for key in raw.info:
            if key not in ['sfreq', 'nchan', 'bads', 'ch_names', 'chs', 'comps']:
                info_downsampled[key] = raw.info[key]

        # 设置电极位置
        if raw.info.get('dig'):
            info_downsampled['dig'] = raw.info['dig']

        # 创建新的Raw对象
        raw_downsampled = RawArray(data_downsampled, info_downsampled)

        downsampling_info['applied'] = True
        downsampling_info['success'] = True

        return raw_downsampled, downsampling_info

    elif method == 'mean_pooling':
        # 均值池化方法
        print(f"  使用均值池化方法降采样...")

        # 先应用抗混叠滤波
        if antialias:
            print(f"  应用抗混叠滤波 (截止频率: {target_sfreq / 2} Hz)...")
            raw_filtered = raw.copy().filter(
                l_freq=None,
                h_freq=target_sfreq / 2 * 0.95,  # 稍微低于Nyquist频率
                fir_design='firwin'
            )
        else:
            raw_filtered = raw.copy()
            print(f"  警告: 均值池化未应用抗混叠滤波，可能出现混叠!")

        # 获取数据
        raw_data = raw_filtered.get_data()
        n_channels, n_samples = raw_data.shape

        # 计算池化后的大小
        pool_size = downsample_factor
        n_pools = n_samples // pool_size

        # 调整数据大小以适应池化
        truncated_samples = n_pools * pool_size
        data_truncated = raw_data[:, :truncated_samples]

        # 重塑并计算均值
        data_reshaped = data_truncated.reshape(n_channels, n_pools, pool_size)
        data_pooled = np.mean(data_reshaped, axis=2)

        # 创建新的Raw对象
        from mne import create_info
        from mne.io import RawArray

        ch_names = raw.info['ch_names']
        ch_types = raw.info.get('chs', [{} for _ in ch_names])
        ch_types = [ch.get('kind', 'eeg') for ch in ch_types]

        info_pooled = create_info(
            ch_names=ch_names,
            sfreq=target_sfreq,
            ch_types=ch_types
        )

        # 复制其他信息
        for key in raw.info:
            if key not in ['sfreq', 'nchan', 'bads', 'ch_names', 'chs', 'comps']:
                info_pooled[key] = raw.info[key]

        # 设置电极位置
        if raw.info.get('dig'):
            info_pooled['dig'] = raw.info['dig']

        raw_pooled = RawArray(data_pooled, info_pooled)

        downsampling_info['applied'] = True
        downsampling_info['success'] = True
        downsampling_info['pool_size'] = pool_size

        return raw_pooled, downsampling_info

    else:
        print(f"  警告: 未知的降采样方法 '{method}'，使用MNE resample作为后备")
        try:
            raw_downsampled = raw.copy().resample(sfreq=target_sfreq, npad='auto')
            downsampling_info['applied'] = True
            downsampling_info['success'] = True
            downsampling_info['method'] = 'resample_fallback'
            return raw_downsampled, downsampling_info
        except Exception as e:
            print(f"  后备方法也失败: {e}, 返回原始数据")
            downsampling_info['applied'] = False
            downsampling_info['success'] = False
            downsampling_info['error'] = str(e)
            return raw.copy(), downsampling_info

def verify_downsampling_results(raw_original, raw_downsampled):
    """
    验证降采样结果

    参数:
    ----------
    raw_original : mne.io.Raw
        原始数据
    raw_downsampled : mne.io.Raw
        降采样后的数据

    返回:
    ----------
    verification_info : dict
        验证信息
    """
    info = {}

    # 基本信息
    info['original_sfreq'] = raw_original.info['sfreq']
    info['downsampled_sfreq'] = raw_downsampled.info['sfreq']
    info['original_shape'] = raw_original.get_data().shape
    info['downsampled_shape'] = raw_downsampled.get_data().shape

    # 计算持续时间
    info['original_duration'] = info['original_shape'][1] / info['original_sfreq']
    info['downsampled_duration'] = info['downsampled_shape'][1] / info['downsampled_sfreq']

    # 降采样因子
    info['downsample_factor'] = info['original_sfreq'] / info['downsampled_sfreq']

    # 数据保留率
    info['data_preservation_ratio'] = info['downsampled_shape'][1] / info['original_shape'][1]

    # 打印验证信息
    print("\n降采样验证信息:")
    print(f"  原始采样率: {info['original_sfreq']} Hz")
    print(f"  降采样后采样率: {info['downsampled_sfreq']} Hz")
    print(f"  降采样因子: {info['downsample_factor']:.2f}")
    print(f"  原始数据形状: {info['original_shape']}")
    print(f"  降采样后形状: {info['downsampled_shape']}")
    print(f"  原始持续时间: {info['original_duration']:.2f} 秒")
    print(f"  降采样后持续时间: {info['downsampled_duration']:.2f} 秒")
    print(f"  数据保留率: {info['data_preservation_ratio']:.2%}")

    return info


def detect_bad_channels_manual(raw, flat_threshold=1e-15, z_threshold=3):
    """手动检测坏通道"""
    import numpy as np
    from scipy import stats

    data = raw.get_data()
    ch_names = raw.ch_names

    # 1. 检测平直通道
    variances = np.var(data, axis=1)
    flat_idx = np.where(variances < flat_threshold)[0]
    flat_channels = [ch_names[i] for i in flat_idx]

    # 2. 检测噪声通道（方差异常）
    # 计算z分数（使用稳健的统计量）
    z_scores = np.abs(stats.zscore(variances))
    noisy_idx = np.where(z_scores > z_threshold)[0]
    noisy_channels = [ch_names[i] for i in noisy_idx
                      if ch_names[i] not in flat_channels]

    # 3. 检测异常峰峰值
    ptp_values = np.ptp(data, axis=1)
    ptp_z_scores = np.abs(stats.zscore(ptp_values))
    extreme_idx = np.where(ptp_z_scores > z_threshold)[0]
    extreme_channels = [ch_names[i] for i in extreme_idx
                        if ch_names[i] not in flat_channels
                        and ch_names[i] not in noisy_channels]

    all_bad = list(set(flat_channels + noisy_channels + extreme_channels))

    if all_bad:
        print(f"手动检测到坏通道: {all_bad}")
        print(f"  - 平直通道: {flat_channels}")
        print(f"  - 噪声通道: {noisy_channels}")
        print(f"  - 异常通道: {extreme_channels}")

    return all_bad