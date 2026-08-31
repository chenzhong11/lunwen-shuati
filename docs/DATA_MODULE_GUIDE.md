# 数据模块使用指南

## 概述

本数据模块提供了完整的 CWRU 轴承数据集处理流程，包括数据加载、预处理、数据集创建和数据泄漏检查。

## 模块结构

```
src/data/
├── __init__.py          # 导出所有公共接口
├── base_loader.py       # 抽象基类
├── cwru_loader.py       # CWRU 数据加载器
├── preprocessing.py     # 数据预处理
├── dataset.py           # PyTorch Dataset
└── leakage_checker.py   # 数据泄漏检查
```

## 快速开始

### 1. 数据加载

```python
from data import CWRULoader

# 配置参数
config = {
    'data_path': './data/cwru',
    'fault_types': ['normal', 'ball_007', 'inner_007'],
    'load_conditions': [0, 1, 2, 3],
    'channel': 'DE',
    'sample_rate': 12000
}

# 创建加载器
loader = CWRULoader(config)

# 下载数据（可选）
loader.download_cwru()

# 加载数据
data = loader.load()
print(f"信号形状: {data['signals'].shape}")
print(f"标签形状: {data['labels'].shape}")

# 划分数据集
splits = loader.get_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### 2. 数据预处理

```python
from data.preprocessing import segment_signal, normalize, preprocess_pipeline

# 信号分段
signal = data['signals'][0]
# 通用 API 示例（legacy/teaching only；正式 Phase 1 使用 2048）
segments = segment_signal(signal, window_length=1024, overlap=0.5)

# 归一化
normalized_data, norm_params = normalize(segments, method='z-score')

# 完整预处理流程
config = {
    'window_length': 1024,  # legacy/teaching only; formal Phase 1 uses 2048
    'overlap': 0.5,
    'normalize_method': 'z-score'
}

processed = preprocess_pipeline(
    data['signals'], 
    data['labels'], 
    data['metadata'],
    config
)
```

### 3. 创建数据集

```python
from data import CWRUDataset, create_scenario, save_dataset

# 创建数据集
dataset = CWRUDataset(signals=processed['signals'], labels=processed['labels'])

# 保存数据集
save_dataset(dataset, './data/processed/train_data.npz')

# 创建跨负载场景的 DataLoader
config = {
    'data_path': './data/processed',
    'batch_size': 32,
    'num_workers': 4,
    'scenario_config': {
        'train_loads': [0, 1],
        'test_loads': [2, 3],
        'val_ratio': 0.2
    }
}

dataloaders = create_scenario(config, scenario='cross_load')
```

### 4. 数据泄漏检查

```python
from data import DataLeakageChecker, quick_leakage_check

# 创建检查器
checker = DataLeakageChecker()

# 检查源域和目标域重叠
result = checker.check_source_target_overlap(
    train_source_info, 
    test_source_info,
    check_type='strict'
)

# 生成报告
report = checker.generate_report('./reports/leakage_report.txt')

# 快速检查
has_leakage = quick_leakage_check(
    train_source_info,
    test_source_info,
    train_params=train_norm_params,
    test_params=test_norm_params,
    report_path='./reports/quick_check.txt'
)
```

## 配置说明

### CWRU 数据集配置

```python
config = {
    'data_path': str,           # 数据存储路径
    'fault_types': list,        # 故障类型列表
    'load_conditions': list,    # 负载条件列表 [0, 1, 2, 3]
    'channel': str,             # 加速度计通道 ('DE', 'FE', 'BA')
    'sample_rate': int          # 采样频率（默认 12kHz）
}
```

### 预处理配置

```python
config = {
    'window_length': int,       # 窗口长度（采样点数）
    'overlap': float,           # 重叠率 (0-1)
    'normalize_method': str,    # 归一化方法 ('z-score', 'min-max', 'robust')
    'normalize_params': dict    # 预计算的归一化参数（可选）
}
```

### 场景配置

```python
config = {
    'data_path': str,           # 数据文件路径
    'batch_size': int,          # 批次大小
    'num_workers': int,         # 数据加载线程数
    'pin_memory': bool,         # 是否使用锁页内存
    'scenario_config': {
        'train_loads': list,    # 训练使用的负载条件
        'test_loads': list,     # 测试使用的负载条件
        'val_ratio': float      # 验证集比例
    }
}
```

## 故障类型映射

| 故障类型 | 描述 | 标签 |
|---------|------|------|
| normal | 正常轴承 | 0 |
| ball_007 | 滚动体故障 (0.007 英寸) | 1 |
| ball_014 | 滚动体故障 (0.014 英寸) | 2 |
| ball_021 | 滚动体故障 (0.021 英寸) | 3 |
| inner_007 | 内圈故障 (0.007 英寸) | 4 |
| inner_014 | 内圈故障 (0.014 英寸) | 5 |
| inner_021 | 内圈故障 (0.021 英寸) | 6 |
| outer_007 | 外圈故障 (0.007 英寸) | 7 |
| outer_014 | 外圈故障 (0.014 英寸) | 8 |
| outer_021 | 外圈故障 (0.021 英寸) | 9 |

## 注意事项

1. **数据泄漏检查**：在跨负载场景中，务必检查训练集和测试集之间是否存在数据泄漏
2. **归一化参数**：测试集应使用训练集的归一化参数，避免数据泄漏
3. **内存使用**：大数据集建议使用生成器或分批加载
4. **PyTorch 依赖**：模块支持有无 PyTorch 两种模式，无 PyTorch 时使用 numpy 实现

## 示例脚本

运行测试脚本验证模块功能：

```bash
cd E:\projects\wdcnn-phm-reproduction
python test_data_module.py
```
