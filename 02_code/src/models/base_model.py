"""
WDCNN 模型基类模块

本模块定义了所有 PHM 模型的抽象基类 BaseModel。
BaseModel 继承自 nn.Module 和 ABC，提供统一的接口规范。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    PHM 模型的抽象基类
    
    所有用于预测与健康管理 (PHM) 的模型都应继承此类。
    提供统一的接口规范，确保模型具有 forward() 和 get_bn_layers() 方法。
    
    Attributes:
        None (子类应定义自己的属性)
    """
    
    def __init__(self) -> None:
        """
        初始化 BaseModel
        
        调用 nn.Module 的构造函数。
        """
        super(BaseModel, self).__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法
        
        子类必须实现此方法，定义模型的前向计算逻辑。
        
        Args:
            x: 输入张量，形状通常为 (batch_size, channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_bn_layers(self) -> List[nn.BatchNorm1d]:
        """
        获取所有 BatchNorm 层的引用
        
        用于域适应 (Domain Adaptation) 等需要访问 BN 层统计量的场景。
        
        Returns:
            包含所有 BatchNorm1d 层引用的列表
        """
        pass
