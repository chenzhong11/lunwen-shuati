"""
create_scenario() 单元测试
==========================

验证跨负载场景创建是否正确，确保无数据泄漏。
"""

import sys
import os
from pathlib import Path
import pytest

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_loader import (
    create_scenario, 
    CWRU_FILE_MAP, 
    FAULT_LABEL_MAP,
    CWRULoader,
    segment_signal,
    normalize_zscore
)
from src.data.leakage_checker import DataLeakageChecker


def test_scenario_s1_to_s6():
    """
    测试 S1-S6 场景创建
    """
    data_path = str(PROJECT_ROOT / "data" / "raw")
    
    mat_files = list(Path(data_path).glob('*.mat'))
    if not mat_files:
        pytest.skip(f"data/raw 不存在可用 .mat 文件: {data_path}")
        
    scenarios = {
        'S1': (1, 2),
        'S2': (1, 3),
        'S3': (2, 3),
        'S4': (2, 1),
        'S5': (3, 1),
        'S6': (3, 2),
    }
    
    all_passed = True
    
    for scenario_name, (source_load, target_load) in scenarios.items():
        print(f"\n--- {scenario_name}: {source_load}HP -> {target_load}HP ---")
        
        try:
            scenario_data = create_scenario(
                data_path=data_path,
                source_load=source_load,
                target_load=target_load,
                window_length=2048,
                source_overlap=0.5,
                target_overlap=0.0,
            )
            
            checker = DataLeakageChecker()
            results = checker.run_all_checks(scenario_data)
            
            print(f"Source load: {scenario_data['source']['load']}")
            print(f"Target load: {scenario_data['target']['load']}")
            print(f"Source samples: {scenario_data['source']['num_samples']}")
            print(f"Target samples: {scenario_data['target']['num_samples']}")
            print(f"Source files: {scenario_data['source']['files']}")
            print(f"Target files: {scenario_data['target']['files']}")
            
            if results['passed']:
                print(f"[PASS] {scenario_name} 泄漏检查通过")
            else:
                print(f"[FAIL] {scenario_name} 泄漏检查失败")
                for check in results['details']['failed']:
                    print(f"   - {check['check']}: {check['detail']}")
                all_passed = False
                
        except Exception as e:
            print(f"[FAIL] {scenario_name} 创建失败: {e}")
            all_passed = False
            
    assert all_passed, "至少一个 S1-S6 场景的泄漏检查失败"


def test_file_mapping():
    """
    测试文件映射是否正确
    """
    print("\n--- 测试文件映射 ---")
    
    for fault_type, load_files in CWRU_FILE_MAP.items():
        for load_hp in [0, 1, 2, 3]:
            if load_hp not in load_files:
                print(f"[FAIL] {fault_type} 缺少 {load_hp}HP 的文件映射")
                assert False, f"{fault_type} 缺少 {load_hp}HP 的文件映射"
                
    print("[PASS] 文件映射完整")


def test_label_mapping():
    """
    测试标签映射是否正确
    """
    print("\n--- 测试标签映射 ---")
    
    if len(FAULT_LABEL_MAP) != 10:
        print(f"[FAIL] 标签映射数量错误: {len(FAULT_LABEL_MAP)} != 10")
        assert len(FAULT_LABEL_MAP) == 10
        
    labels = sorted(FAULT_LABEL_MAP.values())
    if labels != list(range(10)):
        print(f"[FAIL] 标签不连续: {labels}")
        assert labels == list(range(10))
        
    print("[PASS] 标签映射正确")


def test_segment_signal():
    """
    测试信号分段函数
    """
    print("\n--- 测试信号分段 ---")
    
    import numpy as np
    
    signal = np.arange(1000, dtype=float)
    
    # 测试无 overlap
    segments, indices = segment_signal(signal, window_length=100, overlap=0.0)
    assert len(segments) == 10, f"无 overlap 时应有 10 个窗口，实际 {len(segments)}"
    
    # 测试 50% overlap
    segments, indices = segment_signal(signal, window_length=100, overlap=0.5)
    assert len(segments) == 19, f"50% overlap 时应有 19 个窗口，实际 {len(segments)}"
    
    # 验证窗口内容
    assert segments[0][0] == 0, "第一个窗口应从 0 开始"
    assert segments[0][-1] == 99, "第一个窗口应到 99"
    
    print("[PASS] 信号分段正确")


def test_normalize_zscore():
    """
    测试 z-score 归一化
    """
    print("\n--- 测试 z-score 归一化 ---")
    
    import numpy as np
    
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized, mean, std = normalize_zscore(data)
    
    assert abs(np.mean(normalized)) < 1e-10, f"归一化后均值应接近 0"
    assert abs(np.std(normalized) - 1.0) < 1e-10, f"归一化后标准差应接近 1"
    
    print("[PASS] z-score 归一化正确")


def test_random_split_warning():
    """
    验证 get_splits 有警告
    """
    print("\n--- 测试随机划分警告 ---")
    
    import numpy as np
    
    loader = CWRULoader({
        'data_path': 'dummy',
        'fault_types': ['normal'],
    })
    loader.raw_data = {
        'signals': np.zeros((2, 4)),
        'labels': np.array([0, 0]),
    }

    assert hasattr(loader, 'get_splits'), "get_splits 方法应存在"
    with pytest.warns(DeprecationWarning, match='随机划分'):
        loader.get_splits()
    print("[PASS] get_splits 方法存在（已标记为 legacy）")


def test_scenario_requires_real_mat_files(tmp_path):
    """空目录不能被误报为可用的正式场景。"""
    with pytest.raises(FileNotFoundError, match='CWRU .mat'):
        create_scenario(
            data_path=str(tmp_path),
            source_load=1,
            target_load=2,
            window_length=2048,
        )


def _run_manual(name, test_func):
    """为直接运行本文件提供 PASS/FAIL/SKIP 汇总。"""
    try:
        test_func()
        return name, 'PASS'
    except pytest.skip.Exception as exc:
        print(f'[SKIP] {name}: {exc.msg}')
        return name, 'SKIP'
    except Exception as exc:
        print(f'[FAIL] {name}: {exc}')
        return name, 'FAIL'


if __name__ == '__main__':
    import numpy as np
    
    print("="*60)
    print("create_scenario() 单元测试")
    print("="*60)
    
    results = []
    for name, test_func in [
        ('文件映射', test_file_mapping),
        ('标签映射', test_label_mapping),
        ('信号分段', test_segment_signal),
        ('z-score 归一化', test_normalize_zscore),
        ('随机划分警告', test_random_split_warning),
    ]:
        results.append(_run_manual(name, test_func))
    
    # 场景测试（需要实际数据）
    data_path = str(PROJECT_ROOT / "data" / "raw")
    if list(Path(data_path).glob('*.mat')):
        results.append(_run_manual('S1-S6真实数据场景测试', test_scenario_s1_to_s6))
    else:
        results.append(('S1-S6真实数据场景测试', 'SKIP'))
        print(f"\n[SKIP] S1-S6真实数据场景测试：data/raw 不存在可用 .mat 文件")
        
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    counts = {status: 0 for status in ('PASS', 'FAIL', 'SKIP')}
    for name, status in results:
        counts[status] += 1
        print(f"[{status}] {name}")

    print(f"\n{counts['PASS']} PASS")
    print(f"{counts['FAIL']} FAIL")
    print(f"{counts['SKIP']} SKIP")
    if counts['FAIL']:
        print("存在失败的测试")
        raise SystemExit(1)
    if counts['SKIP']:
        print("所有已执行测试通过，但存在未执行测试。")
    else:
        print("所有测试通过。")
