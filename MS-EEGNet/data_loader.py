import pandas as pd
import mne
from pathlib import Path


class EEGDataLoader:
    """数据加载器 - 负责加载EEG数据并转换为MNE格式"""

    def __init__(self, config):
        self.config = config.get('data_loading', {})

    def load_eeg_data(self):
        """加载EEG数据"""
        root_dir = self.config.get('root_dir', "huinao/data")
        sfreq = self.config.get('sfreq', 500)
        selected_samples = self.config.get('selected_samples')

        print("=== 步骤1: 加载EEG数据 ===")
        data_samples = self._load_eeg_data(root_dir, sfreq, selected_samples)

        if not data_samples:
            print("错误: 没有加载到任何数据")
            return None

        return data_samples

    def _load_eeg_data(self, root_dir="huinao/data", sfreq=500, selected_samples=None):
        """遍历脑电数据目录，读取数据并转换为MNE格式"""
        data_samples = []
        root_path = Path(root_dir)

        if not root_path.exists():
            print(f"错误：目录 '{root_dir}' 不存在")
            return data_samples

        # 如果指定了selected_samples，转换为set便于快速查找
        if selected_samples is not None:
            selected_samples_set = set(selected_samples)
            print(f"配置选定的sample: {selected_samples}")
        else:
            selected_samples_set = None
            print("加载所有sample")

        # 第一层：遍历sample目录
        for sample_dir in root_path.iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith('sample'):
                sample_base_name = sample_dir.name

                # 如果指定了selected_samples，只处理选定的sample
                if selected_samples_set is not None and sample_base_name not in selected_samples_set:
                    print(f"跳过sample目录: {sample_base_name} (不在选定列表中)")
                    continue

                print(f"\n处理sample目录: {sample_base_name}")
                sample_found = False

                # 第二层：遍历session目录
                for session_dir in sample_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session'):
                        print(f"  发现session目录: {session_dir.name}")
                        event_file, eeg_file = self._find_data_files(session_dir)

                        if event_file and eeg_file:
                            try:
                                sample_data = {
                                    "sample_name": sample_dir.name,
                                    "session_name": session_dir.name,
                                    "directory": str(session_dir),
                                    "eeg_file" : eeg_file,
                                    "event_file": event_file,
                                }

                                data_samples.append(sample_data)
                                sample_found = True
                            except Exception as e:
                                print(f"    错误: 处理 {session_dir.name} 数据时出错: {e}")
                        else:
                            print(f"    警告: 在 {session_dir.name} 中未找到所需文件")

                if not sample_found:
                    print(f"  警告: 在sample目录 {sample_base_name} 中未找到任何有效session")

        print(f"\n总共处理了 {len(data_samples)} 个session")

        # 打印加载的sample统计
        if selected_samples_set is not None:
            loaded_samples = set([sample['sample_name'].split('_')[0] for sample in data_samples])
            print(f"成功加载的sample: {sorted(loaded_samples)}")
            if len(loaded_samples) < len(selected_samples_set):
                missing_samples = selected_samples_set - loaded_samples
                print(f"警告: 以下sample未找到: {sorted(missing_samples)}")

        return data_samples

    def _find_data_files(self, session_dir):
        """在session目录中查找数据文件"""
        event_file = None
        eeg_file = None

        for file_path in session_dir.iterdir():
            if file_path.is_file():
                if file_path.name == "Event.csv":
                    event_file = file_path
                elif file_path.name == "merged_dataCsv.csv":
                    eeg_file = file_path

        return event_file, eeg_file
