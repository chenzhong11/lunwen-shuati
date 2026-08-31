"""
基础数据加载器抽象模块
====================

定义所有数据加载器必须实现的抽象接口，确保数据加载的一致性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseDataLoader(ABC):
    """
    数据加载器抽象基类
    
    所有数据集加载器都必须继承此类并实现抽象方法。
    提供统一的数据加载和划分接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典，包含数据路径、划分参数等
        """
        self.config = config
        self.data_path = config.get('data_path', './data')
        self.raw_data = None
        self.processed_data = None
        
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        加载原始数据
        
        Returns:
            包含原始数据的字典，格式为:
            {
                'signals': np.ndarray,  # 振动信号数据
                'labels': np.ndarray,   # 标签数据
                'metadata': Dict        # 元数据信息
            }
        """
        pass
    
    @abstractmethod
    def get_splits(
        self, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        将数据划分为训练集、验证集和测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
            random_seed: 随机种子
            
        Returns:
            划分后的数据集字典，格式为:
            {
                'train': (signals, labels),
                'val': (signals, labels),
                'test': (signals, labels)
            }
        """
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据集基本信息
        
        Returns:
            包含数据集统计信息的字典
        """
        if self.raw_data is None:
            return {}
            
        return {
            'num_samples': len(self.raw_data.get('signals', [])),
            'signal_length': self.raw_data['signals'].shape[1] if 'signals' in self.raw_data and len(self.raw_data['signals'].shape) > 1 else 0,
            'num_classes': len(np.unique(self.raw_data.get('labels', []))),
            'class_distribution': dict(zip(*np.unique(self.raw_data.get('labels', []), return_counts=True)))
        }

