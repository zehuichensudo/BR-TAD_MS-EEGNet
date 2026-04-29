import mne
import numpy as np


class DataProcessor:
    """数据处理器 - 处理数据预处理、平衡等操作"""

    def __init__(self, config):
        self.config = config
        self.preprocess_config = config.get('preprocessing', {})
        self.epoch_config = config.get('epoch_creation', {})
        self.label_settings = config.get('label_settings', {})

    def preprocess_data(self, all_samples):
        """预处理所有样本数据"""
        print("\n=== 步骤2: 数据预处理 ===")

        from preprocessing import preprocess_all_samples
        preprocess_all_samples(
            all_samples,
            self.preprocess_config
        )

        return all_samples

    def create_epochs(self, all_samples):
        """创建Epochs"""
        print("\n=== 步骤3: 从预处理数据创建Epochs ===")

        from preprocessing import create_epochs_from_preprocessed_data
        create_epochs_from_preprocessed_data(all_samples, self.epoch_config,self.label_settings)
        return all_samples

    def getdate(self, all_samples):
        # --- 1. 独立循环：提取基础信息、Session IDs 和 原始数据 ---
        print("\n--- 正在同步基础数据与 Session 映射 ---")
        all_data_list = []
        all_session_ids = []
        base_labels = []

        for session_epochs in all_samples:
            # 提取原始数据
            epochs_data = session_epochs['epochs_processed'].get_data(copy=True)
            all_data_list.append(epochs_data)

            # 生成 Session ID
            session_name_id = f"{session_epochs['sample_name']}_{session_epochs['session_name']}"
            num_epochs = len(epochs_data)
            all_session_ids.extend([session_name_id] * num_epochs)

            # 提取基础标签 (用于后续校验)
            base_labels.extend(session_epochs['epochs_processed'].events[:, -1])

        # 转换为 Numpy 数组
        full_raw_data = np.concatenate(all_data_list, axis=0)
        full_session_ids = np.array(all_session_ids)
        full_base_labels = np.array(base_labels)
        print(f"数据预整合完成: 总样本数 {len(full_session_ids)}, 数据形状 {full_raw_data.shape}")
        return full_raw_data, full_session_ids, full_base_labels

