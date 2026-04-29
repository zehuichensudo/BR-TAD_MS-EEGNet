# Artifact-Resilient BCI in Semi-Natural Environments

**Official PyTorch Implementation for the Paper:** *Artifact-Resilient BCI in Semi-Natural Environments: Online Robotic Control via Bidirectional Physical Leakage Decoupling*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english-version) | [中文说明](#chinese-version)

---

<h2 id="english-version">English Version</h2>

### 📖 Introduction

This repository contains the official code for our proposed highly robust Brain-Computer Interface (BCI) online control framework designed for semi-natural environments. It tackles the severe Electromyography (EMG) artifact contamination and the "shortcut learning" pitfalls in deep learning models.

**Key Components:**

1. **BR-TAD (Bidirectional Reciprocal Time-domain Adaptive Decoupling):** A physical leakage decoupling algorithm that isolates high-energy EMG artifacts in the source space without amplitude loss of weak neural rhythms.
2. **MS-EEGNet:** A multi-scale temporal feature reassembly decoding engine. 
3. **Pareto Optimality:** Achieves a **99.01%** classification accuracy with an ultra-low inference latency of **0.43 ms**, proving its global optimality for online robotic control.

### 📊 Datasets

Due to storage limitations, the EEG data is not uploaded directly to this repository. Please download the datasets from the links below and place them in the correct directories for their respective modules.

#### 1. Self-Collected 7-Class Robotic Control Dataset (for MS-EEGNet)
This dataset includes 7-class motor execution signals (Up, Down, Left, Right, Tight, Release, Rest) recorded in a semi-natural environment.
* 📥 **Download Link:** [Google Drive (MS-EEGNet Data)](https://drive.google.com/file/d/1IzcMk6PUDtbqud6gkH2ZLCUdwVH0zOMb/view?usp=sharing)

#### 2. BCI Competition IV 2a Dataset (for BR-TAD & ATCNet Ablation)
Used for the progressive ablation study to evaluate the spatial purification limit of the BR-TAD algorithm.
* 📥 **Download Link:** [BCI Competition IV Official Website](http://www.bbci.de/competition/iv/)

---

### 🛠️ Module 1: BR-TAD-ATCNet

This module evaluates the spatial purification limit of BR-TAD and executes the progressive ablation study.

**1. Environment Setup**
```bash
pip install tensorflow torch numpy scipy scikit-learn matplotlib mne
pip install cupy-cuda11x  # Recommended for GPU acceleration
2. Data Preparation
Create a data folder inside BR-TAD-ATCNet and place the .mat files (s1~s9):

Plaintext
BR-TAD-ATCNet/
├── data/
│   ├── s1/
│   │   ├── A01T.mat
│   │   └── A01E.mat
│   └── ...
3. Run Experiments & Plotting

Bash
# Run the 8-stage progressive ablation study
python train.py

# Generate spatial saliency topomaps (Fig 1, 2, 3 in paper)
python plot_ablation.py

# Verify Mu/Beta rhythm fidelity in the frequency domain
python test_leakage.py
🛠️ Module 2: MS-EEGNet
This module contains the training pipeline for the multi-scale architecture, the system-level Pareto optimality analysis, and spatial explainability visualization.

1. Environment Setup

Bash
pip install torch numpy pandas scipy scikit-learn matplotlib mne seaborn imbalanced-learn
2. Data Preparation
Extract the downloaded 7-class dataset into a DATA folder inside MS-EEGNet:

Plaintext
MS-EEGNet/
├── DATA/
│   ├── sample2/
│   │   ├── session1/
│   │   │   ├── Event.csv
│   │   │   └── merged_dataCsv.csv
│   └── ...
3. Train & Evaluate

Bash
# Step A: Train the Baseline (16ch) and Proposed (10ch + BR-TAD) models
python train.py

# Step B: Run Pareto optimality analysis across 8 models
python train_and_plot_pareto.py

# Step C: Generate XAI interpretability saliency topomaps
python plot_saliency_topomap.py
📝 Citation
If you find this code or our datasets useful in your research, please consider citing our paper:

Code snippet
@article{YourPaper2026,
  title={Artifact-Resilient BCI in Semi-Natural Environments: Online Robotic Control via Bidirectional Physical Leakage Decoupling},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2026}
}
📖 项目简介
本仓库是论文《面向半自然环境的抗伪迹脑机接口：基于双向物理泄漏解耦的机械臂在线控制》的官方 PyTorch 实现代码。本项目致力于解决脑机接口从实验室走向真实环境时面临的强肌电伪迹污染以及深度学习模型的“捷径学习”问题。

核心贡献：

BR-TAD (基于高频锚定的双向互易自适应解耦算法)： 从物理底层解析容积传导的双向耦合，在源空间强制隔离高能肌电伪迹，实现微弱神经节律的无损保真。

MS-EEGNet： 多尺度时间特征重装解码引擎，完美适应肢体与口面部微动作的频带异质性。

帕累托最优效能： 系统在维持 0.43 毫秒极低推理延迟的同时，实现了高达 99.01% 的分类准确率，在精度与速度的双维度对比中确立了工程最优地位。

📊 数据集获取
为了保持代码仓库的轻量化，我们没有将原始 EEG 数据直接上传。请通过以下链接下载数据，并解压至对应模块的目录下。

1. 自采 7 分类机械臂实控数据集 (用于 MS-EEGNet)
该数据集包含了在半自然环境下采集的 7 种运动执行意图（含肢体联动与口面部微动作）。

📥 下载链接: Google Drive 数据集

2. BCI Competition IV 2a 数据集 (用于 BR-TAD 与 ATCNet 消融实验)
用于量化评估 BR-TAD 算法在应对极端频段噪声掩蔽时的空间净化上限。

📥 下载链接: BCI Competition IV 官方网站

🛠️ 模块一运行指南：BR-TAD-ATCNet
该模块包含了 BR-TAD 算法空间净化上限评估 与 ATCNet 基线消融实验 的完整核心代码。所有实验（EXP 1 ~ EXP 8）均已实现自动化。

1. 环境准备

推荐使用 Python 3.8+ 与 GPU 版本的 PyTorch：

Bash
pip install tensorflow torch numpy scipy scikit-learn matplotlib mne
pip install cupy-cuda11x  # 强烈推荐安装 cupy 以开启 GPU 极限加速
2. 数据集放置

在 BR-TAD-ATCNet 目录下新建 data 文件夹，按照受试者放置 .mat 数据：

Plaintext
BR-TAD-ATCNet/
├── data/
│   ├── s1/
│   │   ├── A01T.mat
│   │   └── A01E.mat
│   └── ...
3. 执行大考与绘图

Bash
# 自动执行 8 组递进式消融实验
python train.py

# 读取权重，生成论文所需空间拓扑映射图 (图 1, 图 2, 图 3)
python plot_ablation.py

# 独立核查脑电保真度 (Mu/Beta节律检视)
python test_leakage.py
🛠️ 模块二运行指南：MS-EEGNet
该模块包含了 MS-EEGNet 核心多尺度架构训练、系统级帕累托最优验证与空间可解释性分析代码。

1. 环境准备

Bash
pip install torch numpy pandas scipy scikit-learn matplotlib mne seaborn imbalanced-learn
2. 数据集放置

在 MS-EEGNet 根目录下新建 DATA 文件夹，将下载好的自采数据解压至此：

Plaintext
MS-EEGNet/
├── DATA/
│   ├── sample2/
│   │   ├── session1/
│   │   │   ├── Event.csv
│   │   │   └── merged_dataCsv.csv
│   └── ...
3. 核心训练与模型对比

Bash
# 步骤 A：自动执行 Baseline(16ch) 与 Proposed(10ch + BR-TAD) 的端到端跨场次训练
python train.py

# 步骤 B：一键运行 8 种主流架构在线压测，计算延迟并生成帕累托前沿大图
python train_and_plot_pareto.py

# 步骤 C：执行 XAI 归因，渲染出版级靶心映射图
python plot_saliency_topomap.py
📄 开源协议
本项目采用 MIT License 开源协议。欢迎学术界与工业界同行共同探讨与复现。
