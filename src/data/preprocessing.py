"""
数据预处理模块
=============

提供信号分段、归一化、标签生成等通用预处理功能。
支持用训练集参数归一化测试集，避免数据泄漏。

说明：本模块保留 1024 的通用默认值用于旧示例；正式 Phase 1 WDCNN
不调用该默认流程，而是通过 create_scenario() 使用 2048。
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def segment_signal(
    signal: np.ndarray, 
    window_length: int, 
    overlap: float = 0.5,
    return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """
    将连续信号分段为固定长度的窗口
    
    Args:
        signal: 输入信号数组
        window_length: 窗口长度（采样点数）
        overlap: 重叠率 (0-1)，默认 0.5 表示 50% 重叠
        return_indices: 是否返回每个窗口的起始和结束索引
        
    Returns:
        分段后的信号数组，形状为 (num_windows, window_length)
        如果 return_indices=True，还返回每个窗口的索引列表
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("重叠率必须在 [0, 1) 范围内")
        
    if window_length <= 0:
        raise ValueError("窗口长度必须大于 0")
        
    signal_length = len(signal)
    
    if signal_length < window_length:
        raise ValueError(f"信号长度 ({signal_length}) 小于窗口长度 ({window_length})")
    
    # 计算步长
    step = int(window_length * (1 - overlap))
    if step <= 0:
        step = 1
    
    # 计算窗口数量
    num_windows = (signal_length - window_length) // step + 1
    
    # 提取窗口
    segments = np.zeros((num_windows, window_length))
    indices = []
    
    for i in range(num_windows):
        start = i * step
        end = start + window_length
        segments[i] = signal[start:end]
        indices.append((start, end))
    
    if return_indices:
        return segments, indices
    else:
        return segments


def normalize(
    data: np.ndarray, 
    method: str = 'z-score',
    params: Optional[Dict[str, Any]] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    数据归一化
    
    Args:
        data: 输入数据，形状为 (n_samples, n_features) 或 (n_samples,)
        method: 归一化方法，支持 'z-score', 'min-max', 'robust'
        params: 预计算的归一化参数（用于测试集）
        fit: 是否拟合归一化参数（训练集为 True，测试集为 False）
        
    Returns:
        归一化后的数据和归一化参数
        
    Raises:
        ValueError: 不支持的归一化方法
    """
    if method not in ['z-score', 'min-max', 'robust']:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    # 确保数据是二维的
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    if method == 'z-score':
        if params is not None and not fit:
            # 使用预计算的参数
            mean = params['mean']
            std = params['std']
            normalized = (data - mean) / (std + 1e-8)
        else:
            # 计算新参数
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            normalized = (data - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std}
            
    elif method == 'min-max':
        if params is not None and not fit:
            min_val = params['min']
            max_val = params['max']
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val}
            
    elif method == 'robust':
        if params is not None and not fit:
            median = params['median']
            q75 = params['q75']
            q25 = params['q25']
            iqr = q75 - q25
            normalized = (data - median) / (iqr + 1e-8)
        else:
            median = np.median(data, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            iqr = q75 - q25
            normalized = (data - median) / (iqr + 1e-8)
            params = {'median': median, 'q75': q75, 'q25': q25}
    
    if squeeze_output:
        normalized = normalized.squeeze()
        
    return normalized, params


def create_labels(
    num_samples_per_class: Union[int, List[int]], 
    num_classes: int,
    shuffle: bool = False,
    random_seed: int = 42
) -> np.ndarray:
    """
    创建标签数组
    
    Args:
        num_samples_per_class: 每个类别的样本数，可以是整数（所有类别相同）或列表
        num_classes: 类别数量
        shuffle: 是否打乱标签顺序
        random_seed: 随机种子
        
    Returns:
        标签数组
    """
    if isinstance(num_samples_per_class, int):
        num_samples_list = [num_samples_per_class] * num_classes
    else:
        if len(num_samples_per_class) != num_classes:
            raise ValueError(f"样本数列表长度 ({len(num_samples_per_class)}) 与类别数 ({num_classes}) 不匹配")
        num_samples_list = num_samples_per_class
    
    labels = []
    for class_idx, num_samples in enumerate(num_samples_list):
        labels.extend([class_idx] * num_samples)
    
    labels = np.array(labels)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(labels)
    
    return labels


def create_source_info(
    file_paths: List[str],
    start_positions: List[int],
    load_conditions: List[int],
    window_length: int
) -> List[Dict[str, Any]]:
    """
    创建样本来源信息
    
    Args:
        file_paths: 原始文件路径列表
        start_positions: 每个样本在原始信号中的起始位置
        load_conditions: 负载条件列表
        window_length: 窗口长度
        
    Returns:
        来源信息字典列表
    """
    source_info = []
    
    for i, (file_path, start_pos, load_cond) in enumerate(
        zip(file_paths, start_positions, load_conditions)
    ):
        source_info.append({
            'sample_id': i,
            'file_path': file_path,
            'start_position': start_pos,
            'end_position': start_pos + window_length,
            'load_condition': load_cond,
            'window_length': window_length
        })
    
    return source_info


def preprocess_pipeline(
    signals: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    完整的数据预处理流程
    
    Args:
        signals: 原始信号数据
        labels: 原始标签
        metadata: 元数据列表
        config: 预处理配置，包含:
            - window_length: 窗口长度
            - overlap: 重叠率
            - normalize_method: 归一化方法
            - normalize_params: 预计算的归一化参数（可选）
            
    Returns:
        预处理后的数据字典
    """
    # 通用/legacy API 默认值；正式 Phase 1 由 create_scenario() 显式传入 2048。
    window_length = config.get('window_length', 1024)
    overlap = config.get('overlap', 0.5)
    normalize_method = config.get('normalize_method', 'z-score')
    normalize_params = config.get('normalize_params', None)
    
    # 1. 信号分段
    all_segments = []
    all_segment_labels = []
    all_source_info = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        class_signals = signals[mask]
        class_metadata = [m for m, l in zip(metadata, mask) if l]
        
        for signal_idx, signal in enumerate(class_signals):
            segments, indices = segment_signal(
                signal, window_length, overlap, return_indices=True
            )
            
            all_segments.append(segments)
            all_segment_labels.extend([label] * len(segments))
            
            # 创建来源信息
            for seg_idx, (start, end) in enumerate(indices):
                source_info = {
                    'original_file': class_metadata[signal_idx]['file'] if signal_idx < len(class_metadata) else 'unknown',
                    'start_position': start,
                    'end_position': end,
                    'load_condition': class_metadata[signal_idx].get('load_condition', 0),
                    'fault_type': class_metadata[signal_idx].get('fault_type', 'unknown')
                }
                all_source_info.append(source_info)
    
    # 合并所有分段
    segments = np.vstack(all_segments)
    segment_labels = np.array(all_segment_labels)
    
    # 2. 归一化
    if normalize_params is None:
        # 训练集：计算归一化参数
        normalized_segments, norm_params = normalize(
            segments, method=normalize_method, fit=True
        )
    else:
        # 测试集：使用预计算参数
        normalized_segments, _ = normalize(
            segments, method=normalize_method, params=normalize_params, fit=False
        )
        norm_params = normalize_params
    
    return {
        'signals': normalized_segments,
        'labels': segment_labels,
        'source_info': all_source_info,
        'normalize_params': norm_params,
        'config': {
            'window_length': window_length,
            'overlap': overlap,
            'normalize_method': normalize_method
        }
    }
