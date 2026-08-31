"""
WDCNN (Wide Deep Convolutional Neural Network) 模型实现

本模块实现了论文 "A New Deep Learning Model for Fault Diagnosis with Good
Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals" 中的 WDCNN
架构，用于轴承故障诊断等 PHM 任务。

参考论文 Table 2 的网络结构配置。
论文来源：Zhang et al. (2017), Sensors, 17(2), 425，
DOI: 10.3390/s17020425。
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class WDCNN(BaseModel):
    """
    WDCNN 模型实现
    
    Wide Deep Convolutional Neural Network，专为机械振动信号设计的一维卷积网络。
    特点是第一层使用大卷积核 (64) 和大步长 (16) 来快速提取特征。
    
    网络架构 (论文 Table 2):
        - Conv1: 64×1/stride 16, 16 filters, same padding
        - Pool1: MaxPool 2×1/stride 2
        - Conv2: 3×1/stride 1, 32 filters, same padding
        - Pool2: MaxPool 2×1/stride 2
        - Conv3: 3×1/stride 1, 64 filters, same padding
        - Pool3: MaxPool 2×1/stride 2
        - Conv4: 3×1/stride 1, 64 filters, same padding
        - Pool4: MaxPool 2×1/stride 2
        - Conv5: 3×1/stride 1, 64 filters, valid padding
        - Pool5: MaxPool 2×1/stride 2
        - FC1: 100 neurons
        - Output: num_classes neurons
    
    Args:
        num_classes: 分类类别数，默认为 10 (CWRU 数据集)
        input_length: 输入信号长度，默认为 2048
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_length: int = 2048
    ) -> None:
        """
        初始化 WDCNN 模型
        
        Args:
            num_classes: 分类类别数，默认 10
            input_length: 输入信号长度，默认 2048
        """
        super(WDCNN, self).__init__()
        
        self.num_classes: int = num_classes
        self.input_length: int = input_length
        
        # ==================== 卷积层 ====================
        
        # Conv1: 大卷积核 (64) + 大步长 (16) + same padding
        # 计算 same padding: padding = ((output - 1) * stride + kernel - input) / 2
        # output = ceil(2048 / 16) = 128, padding = 24
        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=64,
            stride=16,
            padding=24  # same padding for input_length=2048
        )
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(16)
        
        # Conv2: 3×1/stride 1, 32 filters, same padding
        self.conv2: nn.Conv1d = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1  # same padding
        )
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(32)
        
        # Conv3: 3×1/stride 1, 64 filters, same padding
        self.conv3: nn.Conv1d = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1  # same padding
        )
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(64)
        
        # Conv4: 3×1/stride 1, 64 filters, same padding
        self.conv4: nn.Conv1d = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1  # same padding
        )
        self.bn4: nn.BatchNorm1d = nn.BatchNorm1d(64)
        
        # Conv5: 3×1/stride 1, 64 filters, valid padding
        self.conv5: nn.Conv1d = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0  # valid padding
        )
        self.bn5: nn.BatchNorm1d = nn.BatchNorm1d(64)
        
        # ==================== 池化层 ====================
        
        # 所有池化层: MaxPool 2×1/stride 2
        self.pool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # ==================== 全连接层 ====================
        
        # 计算卷积后的特征长度
        # 2048 -> conv1(128) -> pool(64) -> conv2(64) -> pool(32)
        # -> conv3(32) -> pool(16) -> conv4(16) -> pool(8)
        # -> conv5(6) -> pool(3)
        self._feature_length: int = self._calculate_feature_length()
        
        # FC1: 100 neurons
        self.fc1: nn.Linear = nn.Linear(64 * self._feature_length, 100)
        self.bn_fc1: nn.BatchNorm1d = nn.BatchNorm1d(100)
        
        # Output layer
        self.fc_out: nn.Linear = nn.Linear(100, num_classes)
        
        # 存储所有 BN 层引用
        self._bn_layers: List[nn.BatchNorm1d] = [
            self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn_fc1
        ]
    
    def _calculate_feature_length(self) -> int:
        """
        计算经过卷积和池化后的特征长度
        
        Returns:
            特征长度
        """
        x: int = self.input_length
        
        # Conv1: stride=16, padding=24, kernel=64
        x = (x + 2 * 24 - 64) // 16 + 1  # 2048 -> 128
        
        # Pool1: kernel=2, stride=2
        x = (x - 2) // 2 + 1  # 128 -> 64
        
        # Conv2: stride=1, padding=1, kernel=3
        x = (x + 2 * 1 - 3) // 1 + 1  # 64 -> 64
        
        # Pool2: kernel=2, stride=2
        x = (x - 2) // 2 + 1  # 64 -> 32
        
        # Conv3: stride=1, padding=1, kernel=3
        x = (x + 2 * 1 - 3) // 1 + 1  # 32 -> 32
        
        # Pool3: kernel=2, stride=2
        x = (x - 2) // 2 + 1  # 32 -> 16
        
        # Conv4: stride=1, padding=1, kernel=3
        x = (x + 2 * 1 - 3) // 1 + 1  # 16 -> 16
        
        # Pool4: kernel=2, stride=2
        x = (x - 2) // 2 + 1  # 16 -> 8
        
        # Conv5: stride=1, padding=0, kernel=3
        x = (x + 2 * 0 - 3) // 1 + 1  # 8 -> 6
        
        # Pool5: kernel=2, stride=2
        x = (x - 2) // 2 + 1  # 6 -> 3
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 1, 2048)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
            
        Example:
            >>> model = WDCNN(num_classes=10)
            >>> x = torch.randn(32, 1, 2048)
            >>> output = model(x)
            >>> output.shape
            torch.Size([32, 10])
        """
        # Block 1: Conv1 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: Conv2 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: Conv3 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4: Conv4 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Block 5: Conv5 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC1 + BN + ReLU
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        # Output layer (logits, softmax in loss)
        x = self.fc_out(x)
        
        return x
    
    def get_bn_layers(self) -> List[nn.BatchNorm1d]:
        """
        获取所有 BatchNorm 层的引用
        
        用于域适应等需要访问 BN 层统计量的场景。
        
        Returns:
            包含所有 BatchNorm1d 层引用的列表
        """
        return self._bn_layers
    
    def get_feature_extractor(self) -> nn.Module:
        """
        获取特征提取器部分（用于迁移学习）
        
        Returns:
            特征提取器
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, model: WDCNN) -> None:
                super(FeatureExtractor, self).__init__()
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.conv2 = model.conv2
                self.bn2 = model.bn2
                self.conv3 = model.conv3
                self.bn3 = model.bn3
                self.conv4 = model.conv4
                self.bn4 = model.bn4
                self.conv5 = model.conv5
                self.bn5 = model.bn5
                self.pool = model.pool
                self.fc1 = model.fc1
                self.bn_fc1 = model.bn_fc1
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = self.pool(F.relu(self.bn4(self.conv4(x))))
                x = self.pool(F.relu(self.bn5(self.conv5(x))))
                x = x.view(x.size(0), -1)
                x = F.relu(self.bn_fc1(self.fc1(x)))
                return x
        
        return FeatureExtractor(self)
