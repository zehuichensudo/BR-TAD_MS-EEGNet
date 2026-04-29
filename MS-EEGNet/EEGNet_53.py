import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_times, F1=16, D=4, dropout=0.5):
        super(EEGNet, self).__init__()

        # 1. 优化多尺度：增加一个中间尺度
        self.ms_block = nn.ModuleList([
            nn.Conv2d(1, F1 // 4, (1, 32), padding=(0, 16), bias=False),  # 极短
            nn.Conv2d(1, F1 // 2, (1, 64), padding=(0, 32), bias=False),  # 中等
            nn.Conv2d(1, F1 // 4, (1, 128), padding=(0, 64), bias=False)  # 较长
        ])
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. 空间卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # 3. 分离卷积
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        self.out_dim = self._get_dim(n_channels, n_times)

        # 4. 分类器优化：增加隐藏层深度
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_dim, 512),  # 增加一层中间过渡
            nn.ELU(),  # 保持与卷积层一致的激活特性
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def _get_dim(self, n_ch, n_t):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_t)
            x = torch.cat([conv(x) for conv in self.ms_block], dim=1)
            x = self.bn1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.ms_block], dim=1)
        x = self.bn1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)