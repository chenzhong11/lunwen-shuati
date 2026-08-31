# 数据模块创建完成

## 已创建的文件

在 `E:\projects\wdcnn-phm-reproduction\src\data\` 目录下成功创建了以下文件：

### 1. `base_loader.py` - 抽象基类
- **类**: `BaseDataLoader(ABC)`
- **抽象方法**:
  - `load()`: 加载原始数据
  - `get_splits()`: 划分训练集、验证集和测试集
- **功能方法**:
  - `get_data_info()`: 获取数据集基本信息

### 2. `cwru_loader.py` - CWRU 数据加载器
- **类**: `CWRULoader(BaseDataLoader)`
- **方法**:
  - `download_cwru(save_dir)`: 下载 CWRU 数据集
  - `load_mat(file_path, channel='DE')`: 加载 .mat 文件
  - `load()`: 加载所有数据
  - `get_splits()`: 划分数据集
- **特性**:
  - 支持故障类型映射 (normal, ball_007, inner_007 等)
  - 支持多负载条件 (0, 1, 2, 3)
  - 支持多通道 (DE, FE, BA)

### 3. `preprocessing.py` - 数据预处理
- **函数**:
  - `segment_signal(signal, window_length, overlap)`: 信号分段
  - `normalize(data, method='z-score', params=None)`: 归一化
  - `create_labels(num_samples_per_class, num_classes)`: 创建标签
  - `create_source_info(...)`: 创建样本来源信息
  - `preprocess_pipeline(...)`: 完整预处理流程
- **归一化方法**: z-score, min-max, robust
- **特性**: 支持用训练集参数归一化测试集

### 4. `dataset.py` - PyTorch Dataset
- **类**: `CWRUDataset(Dataset)`
- **方法**:
  - `__getitem__(idx)`: 获取单个样本
  - `get_source_info(idx)`: 获取样本来源信息
  - `get_class_counts()`: 获取类别分布
- **函数**:
  - `create_scenario(config, scenario)`: 创建跨负载场景
  - `save_dataset(dataset, save_path)`: 保存数据集
  - `load_dataset(data_path)`: 加载数据集
- **场景类型**: cross_load, cross_fault, mixed
- **兼容性**: 支持有无 PyTorch 两种模式

### 5. `leakage_checker.py` - 数据泄漏检查
- **类**: `DataLeakageChecker`
- **方法**:
  - `check_source_target_overlap(source_info, target_info)`: 检查重叠
  - `check_data_split_integrity(...)`: 检查划分完整性
  - `check_normalize_leakage(...)`: 检查归一化参数泄漏
  - `generate_report(save_path)`: 生成报告
- **函数**:
  - `quick_leakage_check(...)`: 快速检查
- **检查类型**: strict, moderate, lenient

### 6. `__init__.py` - 导出所有公共接口
- 导出所有类和函数
- 版本: 1.0.0

## 创建的辅助文件

1. **`test_data_module.py`**: 测试脚本，验证模块功能
2. **`examples/data_module_example.py`**: 完整使用示例
3. **`docs/DATA_MODULE_GUIDE.md`**: 使用指南文档
4. **`config/data_config.yaml`**: 配置文件示例

## 测试结果

所有模块已通过测试：
- ✅ 预处理模块功能正常
- ✅ 数据集创建正常
- ✅ 数据泄漏检查正常
- ✅ 跨负载场景创建正常

## 使用示例

```python
from data import CWRULoader, CWRUDataset, DataLeakageChecker

# 1. 加载数据
config = {
    'data_path': './data/cwru',
    'fault_types': ['normal', 'ball_007', 'inner_007'],
    'load_conditions': [0, 1, 2, 3],
    'channel': 'DE'
}
loader = CWRULoader(config)
data = loader.load()

# 2. 预处理
from data.preprocessing import preprocess_pipeline
processed = preprocess_pipeline(data['signals'], data['labels'], data['metadata'], config)

# 3. 创建数据集
dataset = CWRUDataset(signals=processed['signals'], labels=processed['labels'])

# 4. 检查数据泄漏
checker = DataLeakageChecker()
result = checker.check_source_target_overlap(train_info, test_info)
report = checker.generate_report()
```

## 注意事项

1. **PyTorch 依赖**: 模块支持有无 PyTorch 两种模式，无 PyTorch 时使用 numpy 实现
2. **数据泄漏**: 在跨负载场景中务必检查数据泄漏
3. **归一化**: 测试集应使用训练集的归一化参数
4. **内存使用**: 大数据集建议使用生成器或分批加载
