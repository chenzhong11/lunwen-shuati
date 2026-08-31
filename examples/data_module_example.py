"""
完整使用示例
============

展示数据模块的完整使用流程，包括数据加载、预处理、数据集创建和数据泄漏检查。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import yaml

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def example_data_loading():
    """示例1: 数据加载"""
    from data import CWRULoader
    
    print("=" * 50)
    print("示例1: 数据加载")
    print("=" * 50)
    
    # 配置参数
    config = {
        'data_path': './data/raw/cwru',
        'fault_types': ['normal', 'ball_007', 'inner_007'],
        'load_conditions': [0, 1],
        'channel': 'DE',
        'sample_rate': 12000
    }
    
    # 创建加载器
    loader = CWRULoader(config)
    
    # 下载数据（如果需要）
    # loader.download_cwru()
    
    # 加载数据
    data = loader.load()
    
    # 获取数据信息
    info = loader.get_data_info()
    print(f"样本数量: {info.get('num_samples', 0)}")
    print(f"信号长度: {info.get('signal_length', 0)}")
    print(f"类别数量: {info.get('num_classes', 0)}")
    print(f"类别分布: {info.get('class_distribution', {})}")
    
    # 划分数据集
    splits = loader.get_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"训练集大小: {len(splits['train'][0])}")
    print(f"验证集大小: {len(splits['val'][0])}")
    print(f"测试集大小: {len(splits['test'][0])}")
    
    return data, splits


def example_preprocessing():
    """示例2: 数据预处理"""
    from data.preprocessing import segment_signal, normalize, preprocess_pipeline
    
    print("\n" + "=" * 50)
    print("示例2: 数据预处理")
    print("=" * 50)
    
    # 模拟数据
    signals = np.random.randn(10, 10000)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    metadata = [{'file': f'file_{i}.mat', 'load_condition': i % 4} for i in range(10)]
    
    # 预处理配置
    config = {
        # legacy/teaching example only; formal Phase 1 uses 2048.
        'window_length': 1024,
        'overlap': 0.5,
        'normalize_method': 'z-score'
    }
    
    # 完整预处理流程
    processed = preprocess_pipeline(signals, labels, metadata, config)
    
    print(f"预处理后信号形状: {processed['signals'].shape}")
    print(f"预处理后标签形状: {processed['labels'].shape}")
    print(f"来源信息数量: {len(processed['source_info'])}")
    print(f"归一化参数: {processed['normalize_params'].keys()}")
    
    return processed


def example_dataset_creation():
    """示例3: 数据集创建"""
    from data import CWRUDataset, save_dataset, load_dataset
    
    print("\n" + "=" * 50)
    print("示例3: 数据集创建")
    print("=" * 50)
    
    # 创建模拟数据
    # Synthetic generic example, not a formal WDCNN input fixture.
    signals = np.random.randn(100, 1024)
    labels = np.random.randint(0, 3, 100)
    
    # 创建数据集
    dataset = CWRUDataset(signals=signals, labels=labels)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {dataset.get_num_classes()}")
    print(f"类别分布: {dataset.get_class_counts()}")
    
    # 保存数据集
    save_path = './data/processed/example_dataset.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dataset(dataset, save_path)
    
    # 加载数据集
    loaded_dataset = load_dataset(save_path)
    print(f"加载的数据集大小: {len(loaded_dataset)}")
    
    return dataset


def example_scenario_creation():
    """示例4: 创建跨负载场景"""
    from data import create_scenario
    
    print("\n" + "=" * 50)
    print("示例4: 创建跨负载场景")
    print("=" * 50)
    
    # 创建模拟数据
    signals = np.random.randn(200, 1024)
    labels = np.random.randint(0, 3, 200)
    source_info = [{'load_condition': i % 4} for i in range(200)]
    
    # 保存数据
    save_path = './data/processed/processed_data.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, signals=signals, labels=labels, source_info=source_info)
    
    # 场景配置
    config = {
        'data_path': './data/processed',
        'batch_size': 16,
        'num_workers': 0,
        'scenario_config': {
            'train_loads': [0, 1],
            'test_loads': [2, 3],
            'val_ratio': 0.2
        }
    }
    
    # 创建跨负载场景
    dataloaders = create_scenario(config, scenario='cross_load')
    
    print(f"训练集批次数: {len(dataloaders['train'])}")
    print(f"验证集批次数: {len(dataloaders['val'])}")
    print(f"测试集批次数: {len(dataloaders['test'])}")
    
    # 获取一个批次
    for batch_signals, batch_labels in dataloaders['train']:
        print(f"批次信号形状: {batch_signals.shape}")
        print(f"批次标签形状: {batch_labels.shape}")
        break
    
    return dataloaders


def example_leakage_check():
    """示例5: 数据泄漏检查"""
    from data import DataLeakageChecker, quick_leakage_check
    
    print("\n" + "=" * 50)
    print("示例5: 数据泄漏检查")
    print("=" * 50)
    
    # 模拟来源信息
    train_info = [
        {'file_path': 'train_0.mat', 'start_position': 0, 'end_position': 1024, 'load_condition': 0},
        {'file_path': 'train_0.mat', 'start_position': 512, 'end_position': 1536, 'load_condition': 0},
        {'file_path': 'train_1.mat', 'start_position': 0, 'end_position': 1024, 'load_condition': 1},
    ]
    
    test_info = [
        {'file_path': 'test_2.mat', 'start_position': 0, 'end_position': 1024, 'load_condition': 2},
        {'file_path': 'test_2.mat', 'start_position': 512, 'end_position': 1536, 'load_condition': 2},
        {'file_path': 'test_3.mat', 'start_position': 0, 'end_position': 1024, 'load_condition': 3},
    ]
    
    # 创建检查器
    checker = DataLeakageChecker()
    
    # 检查重叠
    result = checker.check_source_target_overlap(train_info, test_info, check_type='strict')
    print(f"泄漏检测结果: {'发现泄漏' if result['leakage_detected'] else '未发现泄漏'}")
    
    # 生成报告
    report = checker.generate_report('./reports/leakage_check.txt')
    print(f"报告长度: {len(report)} 字符")
    
    # 快速检查
    train_params = {'mean': np.array([0.0]), 'std': np.array([1.0])}
    test_params = {'mean': np.array([0.0]), 'std': np.array([1.0])}
    
    has_leakage = quick_leakage_check(
        train_info, test_info,
        train_params=train_params,
        test_params=test_params,
        report_path='./reports/quick_check.txt'
    )
    print(f"快速检查结果: {'存在泄漏' if has_leakage else '无泄漏'}")
    
    return checker


def main():
    """运行所有示例"""
    print("WDCNN-PHM 数据模块使用示例")
    print("=" * 50)
    
    try:
        # 运行所有示例
        example_data_loading()
        example_preprocessing()
        example_dataset_creation()
        example_scenario_creation()
        example_leakage_check()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


