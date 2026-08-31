"""
PyTorch 数据集模块
=================

提供数据集创建和 DataLoader 生成功能。
支持从 .npz 文件加载预处理数据。
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 尝试导入 PyTorch，如果不可用则使用 numpy 实现
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # 提供基础的 Dataset 和 DataLoader 实现
    class Dataset:
        """基础数据集类"""
        def __init__(self):
            pass
            
        def __len__(self):
            raise NotImplementedError
            
        def __getitem__(self, idx):
            raise NotImplementedError
    
    class DataLoader:
        """基础数据加载器"""
        def __init__(self, dataset, batch_size=32, shuffle=False, num_workers=0, 
                     pin_memory=False, drop_last=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            
        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                np.random.shuffle(indices)
                
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                    
                batch_signals = []
                batch_labels = []
                for idx in batch_indices:
                    signal, label = self.dataset[idx]
                    batch_signals.append(signal)
                    batch_labels.append(label)
                    
                yield np.array(batch_signals), np.array(batch_labels)
                
        def __len__(self):
            n = len(self.dataset) // self.batch_size
            if not self.drop_last and len(self.dataset) % self.batch_size > 0:
                n += 1
            return n


class CWRUDataset(Dataset):
    """
    CWRU 轴承数据集
    
    支持从 .npz 文件加载预处理后的数据，返回 (signal, label) 元组。
    """
    
    def __init__(
        self, 
        data_path: Optional[str] = None,
        signals: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: .npz 文件路径
            signals: 信号数据数组（如果直接提供）
            labels: 标签数组（如果直接提供）
            transform: 信号数据的变换函数
            target_transform: 标签的变换函数
        """
        self.transform = transform
        self.target_transform = target_transform
        
        if data_path is not None:
            # 从文件加载数据
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
                
            data = np.load(data_path, allow_pickle=True)
            self.signals = data['signals']
            self.labels = data['labels']
            
            # 尝试加载来源信息
            self.source_info = data.get('source_info', None)
            
        elif signals is not None and labels is not None:
            # 直接使用提供的数据
            self.signals = signals
            self.labels = labels
            self.source_info = None
            
        else:
            raise ValueError("必须提供 data_path 或 signals 和 labels")
        
        # 确保数据是 numpy 数组
        if not isinstance(self.signals, np.ndarray):
            self.signals = np.array(self.signals)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
            
        # 验证数据形状
        if len(self.signals) != len(self.labels):
            raise ValueError(f"信号数量 ({len(self.signals)}) 与标签数量 ({len(self.labels)}) 不匹配")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (signal, label) 元组
        """
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # 应用变换
        if self.transform is not None:
            signal = self.transform(signal)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        # 转换为 PyTorch 张量（如果可用）
        if PYTORCH_AVAILABLE:
            signal_tensor = torch.FloatTensor(signal)
            label_tensor = torch.LongTensor([label])[0]
            return signal_tensor, label_tensor
        else:
            return signal, label
    
    def get_source_info(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        获取样本的来源信息
        
        Args:
            idx: 样本索引
            
        Returns:
            来源信息字典，如果没有则返回 None
        """
        if self.source_info is not None and idx < len(self.source_info):
            return self.source_info[idx]
        return None
    
    def get_class_counts(self) -> Dict[int, int]:
        """
        获取每个类别的样本数量
        
        Returns:
            类别到数量的映射字典
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))
    
    def get_num_classes(self) -> int:
        """
        获取类别数量
        
        Returns:
            类别数量
        """
        return len(np.unique(self.labels))


def create_scenario(
    config: Dict[str, Any],
    scenario: str = 'cross_load'
) -> Dict[str, DataLoader]:
    """
    创建跨负载场景的 DataLoader
    
    Args:
        config: 配置字典，包含:
            - data_path: 数据文件路径或目录
            - batch_size: 批次大小
            - num_workers: 数据加载线程数
            - pin_memory: 是否使用锁页内存
            - scenario_config: 场景特定配置
        scenario: 场景类型，支持:
            - 'cross_load': 跨负载场景（训练和测试使用不同负载条件）
            - 'cross_fault': 跨故障场景（训练和测试使用不同故障类型）
            - 'mixed': 混合场景
            
    Returns:
        DataLoader 字典，包含 'train', 'val', 'test' 键
    """
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', True) if PYTORCH_AVAILABLE else False
    
    scenario_config = config.get('scenario_config', {})
    
    if scenario == 'cross_load':
        return _create_cross_load_scenario(
            config, scenario_config, batch_size, num_workers, pin_memory
        )
    elif scenario == 'cross_fault':
        return _create_cross_fault_scenario(
            config, scenario_config, batch_size, num_workers, pin_memory
        )
    elif scenario == 'mixed':
        return _create_mixed_scenario(
            config, scenario_config, batch_size, num_workers, pin_memory
        )
    else:
        raise ValueError(f"不支持的场景类型: {scenario}")


def _create_cross_load_scenario(
    config: Dict[str, Any],
    scenario_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Dict[str, DataLoader]:
    """
    创建跨负载场景
    
    Args:
        config: 主配置
        scenario_config: 场景配置，包含:
            - train_loads: 训练使用的负载条件列表
            - test_loads: 测试使用的负载条件列表
            - val_ratio: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用锁页内存
        
    Returns:
        DataLoader 字典
    """
    train_loads = scenario_config.get('train_loads', [0, 1])
    test_loads = scenario_config.get('test_loads', [2, 3])
    val_ratio = scenario_config.get('val_ratio', 0.2)
    
    data_path = config.get('data_path')
    
    # 加载所有数据
    all_data = np.load(os.path.join(data_path, 'processed_data.npz'), allow_pickle=True)
    all_signals = all_data['signals']
    all_labels = all_data['labels']
    all_source_info = all_data.get('source_info', None)
    
    # 根据负载条件划分数据
    if all_source_info is not None:
        train_mask = np.array([
            info['load_condition'] in train_loads 
            for info in all_source_info
        ])
        test_mask = np.array([
            info['load_condition'] in test_loads 
            for info in all_source_info
        ])
    else:
        # 如果没有来源信息，按顺序划分
        n_samples = len(all_signals)
        n_train = int(n_samples * (1 - val_ratio - 0.2))
        n_val = int(n_samples * val_ratio)
        
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
    
    # 提取数据
    train_signals = all_signals[train_mask]
    train_labels = all_labels[train_mask]
    
    test_signals = all_signals[test_mask]
    test_labels = all_labels[test_mask]
    
    # 从训练集中划分验证集
    n_train = len(train_signals)
    n_val = int(n_train * val_ratio)
    
    # 打乱训练数据
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    val_signals = train_signals[val_indices]
    val_labels = train_labels[val_indices]
    
    train_signals = train_signals[train_indices]
    train_labels = train_labels[train_indices]
    
    # 创建数据集
    train_dataset = CWRUDataset(signals=train_signals, labels=train_labels)
    val_dataset = CWRUDataset(signals=val_signals, labels=val_labels)
    test_dataset = CWRUDataset(signals=test_signals, labels=test_labels)
    
    # 创建 DataLoader
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return dataloaders


def _create_cross_fault_scenario(
    config: Dict[str, Any],
    scenario_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Dict[str, DataLoader]:
    """
    创建跨故障场景
    
    Args:
        config: 主配置
        scenario_config: 场景配置
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用锁页内存
        
    Returns:
        DataLoader 字典
    """
    # 类似跨负载场景的实现
    # 这里可以根据需要扩展
    raise NotImplementedError("跨故障场景尚未实现")


def _create_mixed_scenario(
    config: Dict[str, Any],
    scenario_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Dict[str, DataLoader]:
    """
    创建混合场景
    
    Args:
        config: 主配置
        scenario_config: 场景配置
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用锁页内存
        
    Returns:
        DataLoader 字典
    """
    # 随机划分场景
    val_ratio = scenario_config.get('val_ratio', 0.15)
    test_ratio = scenario_config.get('test_ratio', 0.15)
    
    data_path = config.get('data_path')
    
    # 加载数据
    all_data = np.load(os.path.join(data_path, 'processed_data.npz'), allow_pickle=True)
    all_signals = all_data['signals']
    all_labels = all_data['labels']
    
    n_samples = len(all_signals)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # 创建数据集
    train_dataset = CWRUDataset(
        signals=all_signals[train_indices], 
        labels=all_labels[train_indices]
    )
    val_dataset = CWRUDataset(
        signals=all_signals[val_indices], 
        labels=all_labels[val_indices]
    )
    test_dataset = CWRUDataset(
        signals=all_signals[test_indices], 
        labels=all_labels[test_indices]
    )
    
    # 创建 DataLoader
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return dataloaders


def save_dataset(
    dataset: CWRUDataset, 
    save_path: str,
    include_source_info: bool = True
) -> None:
    """
    保存数据集到 .npz 文件
    
    Args:
        dataset: CWRUDataset 实例
        save_path: 保存路径
        include_source_info: 是否包含来源信息
    """
    save_data = {
        'signals': dataset.signals,
        'labels': dataset.labels
    }
    
    if include_source_info and dataset.source_info is not None:
        save_data['source_info'] = dataset.source_info
    
    np.savez(save_path, **save_data)
    print(f"数据集已保存到: {save_path}")


def load_dataset(data_path: str) -> CWRUDataset:
    """
    从 .npz 文件加载数据集
    
    Args:
        data_path: .npz 文件路径
        
    Returns:
        CWRUDataset 实例
    """
    return CWRUDataset(data_path=data_path)

