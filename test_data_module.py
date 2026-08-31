"""
数据模块测试脚本
================

测试数据模块的各个功能。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

def test_preprocessing():
    """测试预处理模块"""
    from data.preprocessing import segment_signal, normalize, create_labels
    
    print("测试预处理模块...")
    
    # 测试信号分段
    signal = np.random.randn(1000)
    segments = segment_signal(signal, window_length=100, overlap=0.5)
    print(f"  信号分段: 输入长度={len(signal)}, 输出形状={segments.shape}")
    
    # 测试归一化
    data = np.random.randn(100, 10)
    normalized, params = normalize(data, method='z-score')
    print(f"  Z-Score归一化: 输入形状={data.shape}, 输出形状={normalized.shape}")
    
    # 测试标签创建
    labels = create_labels([10, 20, 30], num_classes=3)
    print(f"  标签创建: 样本数={len(labels)}, 类别数={len(np.unique(labels))}")
    
    print("预处理模块测试完成!\n")


def test_dataset():
    """测试数据集模块"""
    from data.dataset import CWRUDataset
    
    print("测试数据集模块...")
    
    # 创建模拟数据
    signals = np.random.randn(100, 1000)
    labels = np.random.randint(0, 3, 100)
    
    # 创建数据集
    dataset = CWRUDataset(signals=signals, labels=labels)
    print(f"  数据集大小: {len(dataset)}")
    print(f"  类别数量: {dataset.get_num_classes()}")
    print(f"  类别分布: {dataset.get_class_counts()}")
    
    # 测试获取样本
    signal, label = dataset[0]
    print(f"  样本形状: {signal.shape}, 标签: {label}")
    
    print("数据集模块测试完成!\n")


def test_leakage_checker():
    """测试数据泄漏检查模块"""
    from data.leakage_checker import DataLeakageChecker
    
    print("测试数据泄漏检查模块...")
    
    checker = DataLeakageChecker()
    
    # 模拟来源信息
    source_info = [
        {'file_path': 'file1.mat', 'start_position': 0, 'end_position': 100, 'load_condition': 0},
        {'file_path': 'file1.mat', 'start_position': 50, 'end_position': 150, 'load_condition': 0},
    ]
    
    target_info = [
        {'file_path': 'file2.mat', 'start_position': 0, 'end_position': 100, 'load_condition': 2},
        {'file_path': 'file2.mat', 'start_position': 100, 'end_position': 200, 'load_condition': 2},
    ]
    
    # 检查重叠
    result = checker.check_source_target_overlap(source_info, target_info)
    print(f"  泄漏检测: {'发现泄漏' if result['leakage_detected'] else '未发现泄漏'}")
    
    # 生成报告
    report = checker.generate_report()
    print(f"  报告长度: {len(report)} 字符")
    
    print("数据泄漏检查模块测试完成!\n")


def test_cwru_loader():
    """测试 CWRU 加载器（不实际下载）"""
    from data.cwru_loader import CWRULoader
    
    print("测试 CWRU 加载器...")
    
    config = {
        'data_path': './data/cwru',
        'fault_types': ['normal', 'ball_007', 'inner_007'],
        'load_conditions': [0, 1],
        'channel': 'DE'
    }
    
    loader = CWRULoader(config)
    print(f"  配置加载成功: {loader.fault_types}")
    print(f"  负载条件: {loader.load_conditions}")
    print(f"  通道: {loader.channel}")
    
    print("CWRU 加载器测试完成!\n")


def main():
    """运行所有测试"""
    print("=" * 50)
    print("数据模块功能测试")
    print("=" * 50)
    print()
    
    try:
        test_preprocessing()
        test_dataset()
        test_leakage_checker()
        test_cwru_loader()
        
        print("=" * 50)
        print("所有测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
