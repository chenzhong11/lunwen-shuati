"""
域适应适配器抽象基类模块

本模块定义了所有域适应方法的抽象接口。
所有域适应算法（如 AdaBN、TCA 等）都应继承此基类并实现其抽象方法。

论文引用：
- Li et al., "Deep Domain Generalization via Conditional Invariant Adversarial Networks", ECCV 2018
- 原始 AdaBN 论文：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016 Workshop
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAdapter(ABC):
    """
    域适应适配器的抽象基类
    
    所有域适应方法都应继承此类并实现其抽象方法。
    提供统一的接口规范，确保域适应算法具有 collect_source_stats()、
    compute_target_stats() 和 apply_adaptation() 三个核心方法。
    
    Attributes:
        model: 待适应的神经网络模型
        is_adapted: 是否已完成适应
    """
    
    def __init__(self, model: nn.Module) -> None:
        """
        初始化 BaseAdapter
        
        Args:
            model: 待适应的神经网络模型，通常为 WDCNN 等 PHM 模型
        """
        self.model = model
        self.is_adapted = False
    
    @abstractmethod
    def collect_source_stats(self) -> Dict[str, Any]:
        """
        收集源域训练后的统计量
        
        在源域训练完成后调用，收集模型在源域上学到的统计量。
        对于 AdaBN，这些统计量包括 BatchNorm 层的 running_mean、running_var、
        gamma（缩放参数）和 beta（偏移参数）。
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016
        
        Returns:
            包含源域统计量的字典，具体格式由子类定义
        """
        pass
    
    @abstractmethod
    def compute_target_stats(self, target_loader: DataLoader) -> Dict[str, Any]:
        """
        计算目标域的统计量
        
        使用目标域数据（无标签）计算新的统计量。
        论文明确指出：应使用目标域所有样本计算统计量，而非 mini-batch。
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016, Algorithm 1
        
        Args:
            target_loader: 目标域数据加载器，数据应为无标签的（仅特征）
            
        Returns:
            包含目标域统计量的字典，具体格式由子类定义
        """
        pass
    
    @abstractmethod
    def apply_adaptation(self) -> None:
        """
        应用域适应
        
        将计算得到的目标域统计量应用到模型上，完成域适应过程。
        对于 AdaBN，此方法会用目标域统计量替换 BatchNorm 层的 running_mean 和 running_var，
        同时保持 gamma 和 beta 参数不变。
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016, Algorithm 1
        
        Raises:
            RuntimeError: 如果在调用此方法前未完成 collect_source_stats() 和 compute_target_stats()
        """
        pass
    
    @abstractmethod
    def evaluate(self, target_loader: DataLoader) -> Dict[str, float]:
        """
        使用适应后的模型在目标域上评估
        
        在完成域适应后，使用适应后的模型在目标域数据上进行评估。
        评估指标通常包括准确率、损失等。
        
        Args:
            target_loader: 目标域数据加载器（应包含标签用于评估）
            
        Returns:
            包含评估指标的字典，如 {'accuracy': 0.95, 'loss': 0.1}
        """
        pass
