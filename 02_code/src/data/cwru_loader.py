"""
CWRU 数据集加载器
================

实现 Case Western Reserve University 轴承数据集的下载、加载和处理。
支持多种故障类型和负载条件。

重要：Phase 1 实验必须使用 create_scenario() 创建跨负载场景，
      禁止使用 get_splits() 的随机划分逻辑。
"""

import os
import warnings
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from .base_loader import BaseDataLoader


# ============================================================
# CWRU 文件映射表
# ============================================================
# 来源：CWRU Bearing Data Center
# https://engineering.case.edu/bearingdatacenter
#
# 每个故障类型在不同负载下的 .mat 文件编号
# 格式：{fault_type: {load_hp: filename}}
# ============================================================

CWRU_FILE_MAP = {
    'normal': {
        0: '97.mat',
        1: '98.mat',
        2: '99.mat',
        3: '100.mat',
    },
    'ball_007': {
        0: '118.mat',
        1: '119.mat',
        2: '120.mat',
        3: '121.mat',
    },
    'ball_014': {
        0: '185.mat',
        1: '186.mat',
        2: '187.mat',
        3: '188.mat',
    },
    'ball_021': {
        0: '222.mat',
        1: '223.mat',
        2: '224.mat',
        3: '225.mat',
    },
    'inner_007': {
        0: '105.mat',
        1: '106.mat',
        2: '107.mat',
        3: '108.mat',
    },
    'inner_014': {
        0: '169.mat',
        1: '170.mat',
        2: '171.mat',
        3: '172.mat',
    },
    'inner_021': {
        0: '209.mat',
        1: '210.mat',
        2: '211.mat',
        3: '212.mat',
    },
    'outer_007': {
        0: '130.mat',
        1: '131.mat',
        2: '132.mat',
        3: '133.mat',
    },
    'outer_014': {
        0: '197.mat',
        1: '198.mat',
        2: '199.mat',
        3: '200.mat',
    },
    'outer_021': {
        0: '234.mat',
        1: '235.mat',
        2: '236.mat',
        3: '237.mat',
    },
}

# 故障类型到整数标签的映射
FAULT_LABEL_MAP = {
    'normal': 0,
    'ball_007': 1,
    'ball_014': 2,
    'ball_021': 3,
    'inner_007': 4,
    'inner_014': 5,
    'inner_021': 6,
    'outer_007': 7,
    'outer_014': 8,
    'outer_021': 9,
}


class CWRULoader(BaseDataLoader):
    """
    CWRU 轴承数据集加载器
    
    支持下载、加载和处理 CWRU 轴承数据集，包括正常和多种故障类型。
    
    重要：
    - Phase 1 实验必须使用 create_scenario() 创建跨负载场景
    - get_splits() 已标记为 legacy，仅用于非论文实验
    """
    
    # CWRU 数据集 URL 配置
    BASE_URL = "https://engineering.case.edu/sites/default/files/"
    
    # 故障类型映射
    FAULT_TYPES = {
        'normal': 'Normal',
        'ball_007': 'Ball_0.007',
        'ball_014': 'Ball_0.014', 
        'ball_021': 'Ball_0.021',
        'inner_007': 'Inner_0.007',
        'inner_014': 'Inner_0.014',
        'inner_021': 'Inner_0.021',
        'outer_007': 'Outer_0.007',
        'outer_014': 'Outer_0.014',
        'outer_021': 'Outer_0.021'
    }
    
    # 通道映射
    CHANNEL_MAP = {
        'DE': 'DE',  # Drive End accelerometer
        'FE': 'FE',  # Fan End accelerometer
        'BA': 'BA'   # Base accelerometer
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 CWRU 数据加载器
        
        Args:
            config: 配置字典，包含:
                - data_path: 数据存储路径
                - fault_types: 需要加载的故障类型列表
                - channel: 加速度计通道 ('DE', 'FE', 'BA')
                - sample_rate: 采样频率 (默认 12kHz)
        """
        super().__init__(config)
        self.fault_types = config.get('fault_types', list(self.FAULT_TYPES.keys()))
        self.channel = config.get('channel', 'DE')
        self.sample_rate = config.get('sample_rate', 12000)
        
    def download_cwru(self, save_dir: Optional[str] = None) -> bool:
        """
        下载 CWRU 轴承数据集
        
        Args:
            save_dir: 保存目录，默认使用配置中的 data_path
            
        Returns:
            下载是否成功
        """
        if save_dir is None:
            save_dir = self.data_path
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建下载 URL 列表
        download_urls = []
        
        for fault_type, load_files in CWRU_FILE_MAP.items():
            if fault_type in self.fault_types:
                for load_hp, filename in load_files.items():
                    url = f"{self.BASE_URL}{filename}"
                    filepath = os.path.join(save_dir, filename)
                    if not os.path.exists(filepath):
                        download_urls.append((url, filepath))
        
        # 执行下载
        success_count = 0
        for url, filepath in download_urls:
            try:
                print(f"下载: {url} -> {filepath}")
                urllib.request.urlretrieve(url, filepath)
                success_count += 1
            except Exception as e:
                print(f"下载失败 {url}: {e}")
                
        print(f"下载完成: {success_count}/{len(download_urls)} 个文件")
        return success_count == len(download_urls)
    
    def load_mat(self, file_path: str, channel: Optional[str] = None) -> np.ndarray:
        """
        加载 .mat 文件并提取指定通道的振动信号
        
        Args:
            file_path: .mat 文件路径
            channel: 加速度计通道，如果为 None 则使用默认通道
            
        Returns:
            振动信号数组
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 指定通道不存在
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        if channel is None:
            channel = self.channel
            
        # 加载 .mat 文件
        mat_data = loadmat(file_path)
        
        # 提取振动信号
        # CWRU 数据格式: X___DE_time, X___FE_time, X___BA_time
        signal_key = None
        for key in mat_data.keys():
            if channel in key and 'time' in key:
                signal_key = key
                break
                
        if signal_key is None:
            # 尝试其他可能的键名格式
            possible_keys = [
                f"X{channel}_time",
                f"{channel}_time", 
                channel,
                "signal",
                "data"
            ]
            for key in possible_keys:
                if key in mat_data:
                    signal_key = key
                    break
                    
        if signal_key is None:
            raise ValueError(f"在 {file_path} 中找不到通道 {channel} 的数据")
            
        signal = mat_data[signal_key].flatten()
        return signal
    
    def load_load_condition(self, load_hp: int) -> Dict[str, np.ndarray]:
        """
        加载指定负载条件下的所有故障类型数据
        
        Args:
            load_hp: 负载条件 (0, 1, 2, 3)
            
        Returns:
            字典，键为故障类型，值为振动信号数组
        """
        signals = {}
        
        for fault_type in self.fault_types:
            if fault_type not in CWRU_FILE_MAP:
                continue
                
            if load_hp not in CWRU_FILE_MAP[fault_type]:
                continue
                
            filename = CWRU_FILE_MAP[fault_type][load_hp]
            file_path = os.path.join(self.data_path, filename)
            
            if os.path.exists(file_path):
                signal = self.load_mat(file_path)
                signals[fault_type] = signal
            else:
                print(f"警告: 文件不存在 {file_path}")
                
        return signals
    
    def load(self) -> Dict[str, Any]:
        """
        加载 CWRU 数据集（所有负载条件）
        
        Returns:
            包含所有数据的字典
        """
        all_signals = []
        all_labels = []
        all_metadata = []
        
        for load_hp in [0, 1, 2, 3]:
            signals = self.load_load_condition(load_hp)
            
            for fault_type, signal in signals.items():
                label = FAULT_LABEL_MAP.get(fault_type, -1)
                if label == -1:
                    continue
                    
                all_signals.append(signal)
                all_labels.extend([label] * len(signal))
                all_metadata.extend([{
                    'file': CWRU_FILE_MAP[fault_type][load_hp],
                    'fault_type': fault_type,
                    'load_condition': load_hp,
                    'position': i
                } for i in range(len(signal))])
        
        self.raw_data = {
            'signals': np.concatenate(all_signals) if all_signals else np.array([]),
            'labels': np.array(all_labels),
            'metadata': all_metadata
        }
        
        return self.raw_data
    
    def get_splits(
        self, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        [LEGACY] 将数据划分为训练集、验证集和测试集
        
        ⚠️ 警告：此方法使用随机划分，不符合论文跨负载评估要求。
           Phase 1 实验禁止使用此方法，请使用 create_scenario()。
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
            random_seed: 随机种子
            
        Returns:
            划分后的数据集字典
        """
        warnings.warn(
            "get_splits() 使用随机划分，不符合论文跨负载评估要求。"
            "Phase 1 实验请使用 create_scenario() 方法。",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.raw_data is None:
            self.load()
            
        signals = self.raw_data['signals']
        labels = self.raw_data['labels']
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 获取唯一类别
        unique_labels = np.unique(labels)
        
        train_signals, train_labels = [], []
        val_signals, val_labels = [], []
        test_signals, test_labels = [], []
        
        # 按类别分层划分
        for label in unique_labels:
            mask = labels == label
            class_signals = signals[mask]
            
            n_samples = len(class_signals)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 打乱索引
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            # 划分数据
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            train_signals.extend(class_signals[train_idx])
            train_labels.extend([label] * len(train_idx))
            
            val_signals.extend(class_signals[val_idx])
            val_labels.extend([label] * len(val_idx))
            
            test_signals.extend(class_signals[test_idx])
            test_labels.extend([label] * len(test_idx))
        
        return {
            'train': (np.array(train_signals), np.array(train_labels)),
            'val': (np.array(val_signals), np.array(val_labels)),
            'test': (np.array(test_signals), np.array(test_labels))
        }


def segment_signal(
    signal: np.ndarray,
    window_length: int,
    overlap: float = 0.0
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    将连续信号分段为固定长度的窗口
    
    Args:
        signal: 输入信号
        window_length: 窗口长度
        overlap: 重叠率 (0-1)
        
    Returns:
        分段后的信号数组和每个窗口的 (start, end) 索引
    """
    step = int(window_length * (1 - overlap))
    if step <= 0:
        step = 1
        
    num_windows = (len(signal) - window_length) // step + 1
    
    segments = np.zeros((num_windows, window_length))
    indices = []
    
    for i in range(num_windows):
        start = i * step
        end = start + window_length
        segments[i] = signal[start:end]
        indices.append((start, end))
        
    return segments, indices


def normalize_zscore(data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    z-score 归一化
    
    Args:
        data: 输入数据
        mean: 预计算的均值（用于测试集）
        std: 预计算的标准差（用于测试集）
        
    Returns:
        归一化后的数据、均值、标准差
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
        
    if std == 0:
        std = 1
        
    normalized = (data - mean) / std
    return normalized, mean, std


def create_scenario(
    data_path: str,
    source_load: int,
    target_load: int,
    window_length: int = 2048,
    source_overlap: float = 0.5,
    target_overlap: float = 0.0,
    channel: str = 'DE',
    fault_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    创建跨负载场景（论文 Table 3）
    
    严格按照论文 Section 4.4.1 实现：
    - source_load: 源域负载条件
    - target_load: 目标域负载条件
    - source 和 target 来自不同的原始 .mat 文件
    - 训练集使用 overlap 增强，测试集无 overlap
    
    Args:
        data_path: CWRU 数据目录
        source_load: 源域负载 (1, 2, 3)
        target_load: 目标域负载 (1, 2, 3)
        window_length: 窗口长度 (默认 2048)
        source_overlap: 源域训练集 overlap (默认 0.5)
        target_overlap: 目标域测试集 overlap (默认 0.0)
        channel: 加速度计通道 (默认 'DE')
        fault_types: 故障类型列表
        
    Returns:
        包含 source 和 target 数据的字典
        
    Raises:
        ValueError: source_load == target_load
    """
    if source_load == target_load:
        raise ValueError(f"source_load ({source_load}) 不能等于 target_load ({target_load})")
        
    if fault_types is None:
        fault_types = list(FAULT_LABEL_MAP.keys())
        
    # 初始化加载器
    loader = CWRULoader({
        'data_path': data_path,
        'fault_types': fault_types,
        'channel': channel,
    })

    # 正式 Phase 1 需要每个请求类别在两个域都存在。提前失败，避免
    # 空目录或部分数据被误报为“场景创建成功”。
    missing_files = []
    for load_hp, domain_name in (
        (source_load, 'source'),
        (target_load, 'target'),
    ):
        for fault_type in fault_types:
            filename = CWRU_FILE_MAP[fault_type][load_hp]
            if not os.path.exists(os.path.join(data_path, filename)):
                missing_files.append(f'{domain_name}:{filename}')
    if missing_files:
        preview = ', '.join(missing_files[:6])
        suffix = ' ...' if len(missing_files) > 6 else ''
        raise FileNotFoundError(
            '缺少正式场景所需的 CWRU .mat 文件: '
            f'{preview}{suffix}。请先将原始数据放入 {data_path}'
        )

    # 加载源域和目标域数据
    source_signals = loader.load_load_condition(source_load)
    target_signals = loader.load_load_condition(target_load)
    
    # 处理源域数据（训练集，使用 overlap）
    source_data = []
    source_labels = []
    source_provenance = []
    source_files = set()
    
    for fault_type, signal in source_signals.items():
        label = FAULT_LABEL_MAP.get(fault_type, -1)
        if label == -1:
            continue
            
        filename = CWRU_FILE_MAP[fault_type][source_load]
        source_files.add(filename)
        
        # 分段
        segments, indices = segment_signal(signal, window_length, source_overlap)
        
        # 归一化
        segments, mean, std = normalize_zscore(segments)
        
        source_data.append(segments)
        source_labels.extend([label] * len(segments))
        
        # 记录 provenance
        for start, end in indices:
            source_provenance.append({
                'source_file': filename,
                'load': source_load,
                'fault_type': fault_type,
                'fault_class': label,
                'window_start': start,
                'window_end': end,
            })
    
    # 处理目标域数据（测试集，无 overlap）
    target_data = []
    target_labels = []
    target_provenance = []
    target_files = set()
    
    for fault_type, signal in target_signals.items():
        label = FAULT_LABEL_MAP.get(fault_type, -1)
        if label == -1:
            continue
            
        filename = CWRU_FILE_MAP[fault_type][target_load]
        target_files.add(filename)
        
        # 分段
        segments, indices = segment_signal(signal, window_length, target_overlap)
        
        # 归一化（使用目标域自己的统计量，因为 AdaBN 会重新计算）
        segments, mean, std = normalize_zscore(segments)
        
        target_data.append(segments)
        target_labels.extend([label] * len(segments))
        
        # 记录 provenance
        for start, end in indices:
            target_provenance.append({
                'source_file': filename,
                'load': target_load,
                'fault_type': fault_type,
                'fault_class': label,
                'window_start': start,
                'window_end': end,
            })
    
    # 拼接数据
    source_X = np.concatenate(source_data) if source_data else np.array([])
    source_y = np.array(source_labels)
    target_X = np.concatenate(target_data) if target_data else np.array([])
    target_y = np.array(target_labels)
    
    # 转换为 PyTorch 格式 (N, 1, L)
    if len(source_X) > 0:
        source_X = source_X.reshape(-1, 1, window_length)
    if len(target_X) > 0:
        target_X = target_X.reshape(-1, 1, window_length)
    
    return {
        'source': {
            'X': source_X,
            'y': source_y,
            'provenance': source_provenance,
            'files': source_files,
            'load': source_load,
            'num_samples': len(source_y),
        },
        'target': {
            'X': target_X,
            'y': target_y,
            'provenance': target_provenance,
            'files': target_files,
            'load': target_load,
            'num_samples': len(target_y),
        },
        'config': {
            'window_length': window_length,
            'source_overlap': source_overlap,
            'target_overlap': target_overlap,
            'channel': channel,
            'fault_types': fault_types,
        }
    }
