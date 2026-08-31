"""
运行单个跨负载场景

使用 create_scenario() 创建场景数据，确保无数据泄漏。
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_scenario, DataLeakageChecker


def main():
    parser = argparse.ArgumentParser(description='运行单个跨负载场景')
    parser.add_argument('--scenario', type=str, required=True, 
                        choices=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                        help='场景编号')
    parser.add_argument('--method', type=str, required=True,
                        choices=['wdcnn', 'wdcnn_adabn'],
                        help='方法: wdcnn (baseline) 或 wdcnn_adabn')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='配置文件')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--data-path', type=str, default='data/raw', help='数据目录')
    parser.add_argument('--output', type=str, default='experiments/results', help='输出目录')
    args = parser.parse_args()
    
    # 场景映射
    SCENARIO_MAP = {
        'S1': (1, 2),  # 1HP → 2HP
        'S2': (1, 3),  # 1HP → 3HP
        'S3': (2, 3),  # 2HP → 3HP
        'S4': (2, 1),  # 2HP → 1HP
        'S5': (3, 1),  # 3HP → 1HP
        'S6': (3, 2),  # 3HP → 2HP
    }
    
    source_load, target_load = SCENARIO_MAP[args.scenario]
    
    print(f"场景: {args.scenario}")
    print(f"源域: {source_load}HP")
    print(f"目标域: {target_load}HP")
    print(f"方法: {args.method}")
    print(f"种子: {args.seed}")
    print(f"时间: {datetime.now().isoformat()}")
    
    # 创建场景数据
    print("\n创建场景数据...")
    data_path = str(PROJECT_ROOT / args.data_path)
    
    scenario_data = create_scenario(
        data_path=data_path,
        source_load=source_load,
        target_load=target_load,
        window_length=2048,
        source_overlap=0.5,
        target_overlap=0.0,
    )
    
    # 运行泄漏检查
    print("\n运行泄漏检查...")
    checker = DataLeakageChecker()
    results = checker.run_all_checks(scenario_data)
    
    if not results['passed']:
        print("❌ 数据泄漏检查失败，终止实验")
        for check in results['details']['failed']:
            print(f"   - {check['check']}: {check['detail']}")
        return
        
    print("✅ 数据泄漏检查通过")
    
    # 输出数据统计
    print(f"\n数据统计:")
    print(f"  源域样本数: {scenario_data['source']['num_samples']}")
    print(f"  目标域样本数: {scenario_data['target']['num_samples']}")
    print(f"  源域文件: {scenario_data['source']['files']}")
    print(f"  目标域文件: {scenario_data['target']['files']}")
    
    # TODO: 实现训练和评估逻辑
    print("\n⚠️ 训练和评估逻辑待实现")


if __name__ == '__main__':
    main()
