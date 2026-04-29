# config.py
# 脑机接口（BCI）实时系统 - 核心配置文件
# 修改提示：如果更换了实验范式或受试者，请优先调整下方的数据来源

CONFIG = {
    # 1. 深度学习模型训练相关的超参数
    'deep_learning': {
        'num_epochs': 500,  # 最大训练多少轮
        'patience': 30,   # 早停机制：验证集要是100轮都不进步，就赶紧停，省得过拟合
        'batch_size': 32,  # 每次喂给模型的数据量
        'learning_rate': 5e-4 # 初始学习率，这里设得比较稳
    },

    # 2. 实验范式的标签映射

    'label_settings': {
        'fixed_mapping': {
            'ImageStart_up': 0,
            'ImageStart_down': 1,
            'ImageStart_left': 2,
            'ImageStart_right': 3,
            'ImageStart_tight': 4,
            'ImageStart_release': 5,
            'ImageStart_rest': 6,
        }
    },

    # 3. 数据来源及原始格式
    'data_loading': {
        'root_dir': "./DATA",      # 原始脑电数据存放路径
        'sfreq': 500,              # 脑电帽的原始采样率
        'selected_samples': ["sample2"] # 选取的样本文件夹名称
    },

    # 4. 数据预处理与切分策略
    "data_selection": {
        "split_mode": "session",   # 按实验场次切分测试集，比随机切分更符合真实测试场景
        "test_size": 0.1,          # 拿10%的数据出来做最终大考

        # 降采样配置：500Hz转125Hz

        "resample_config": {
            "enabled": True,       # 是否开启降采样
            "original_fs": 500,    # 输入频率
            "target_fs": 125       # 目标输出频率
        },

        # 滑动窗口配置
        'window_config': {
            'enabled': True,
            'window_size': 2000,  # 4.0秒
            'step': 50,
        },

        "use_ea": False,  # 欧几里得对齐
    },

    # 5. 预处理配置
    'preprocessing': {
        'l_freq': 0.5,
        'h_freq': 100,
        'notch_filter': True,
        'notch_freqs': 50,
        'use_ica': False,
    },

    # BR-TAD 专属配置
    'br_tad_config': {
        'enabled': True,
        'fs': 500,
        # 这里先占位，稍后会在 train.py 中动态获取 MNE 的真实通道名，防止错乱
        'all_channels': [],
        'edge_channels': ['F3', 'F4', 'T7', 'T8', 'O1', 'O2'],
        'center_channels': ['C3', 'C1', 'Cz', 'C2', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'Fz'],
    },

    # 6. Epoch 切分配置
    'epoch_creation': {
        'tmin': 0,
        'tmax': 5.0,
        'tmin_crop': 0,
        'tmax_crop': 5,
        'baseline': None
    }

}