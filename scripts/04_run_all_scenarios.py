"""
运行全部 6 个跨负载场景

使用 create_scenario() 创建场景数据，确保无数据泄漏。
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_scenario, DataLeakageChecker


def main():
    parser = argparse.ArgumentParser(description='检查全部跨负载场景的数据协议（不训练）')
    parser.add_argument('--method', type=str, required=True,
                        choices=['wdcnn', 'wdcnn_adabn', 'both'],
                        help='仅保留兼容参数；当前只做场景/泄漏检查，不训练')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='配置文件')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--data-path', type=str, default='data/raw', help='数据目录')
    parser.add_argument('--output', type=str, default='experiments/results', help='输出目录')
    args = parser.parse_args()
    
    # S1-S6 场景定义
    SCENARIOS = {
        'S1': (1, 2),  # 1HP → 2HP
        'S2': (1, 3),  # 1HP → 3HP
        'S3': (2, 3),  # 2HP → 3HP
        'S4': (2, 1),  # 2HP → 1HP
        'S5': (3, 1),  # 3HP → 1HP
        'S6': (3, 2),  # 3HP → 2HP
    }
    
    methods = ['wdcnn', 'wdcnn_adabn'] if args.method == 'both' else [args.method]
    
    print(f"运行 {len(SCENARIOS)} 个场景 × {len(methods)} 种方法")
    print(f"种子: {args.seed}")
    print(f"时间: {datetime.now().isoformat()}")
    
    data_path = str(PROJECT_ROOT / args.data_path)
    
    ready_count = 0
    failed_count = 0
    not_ready_count = 0

    # 仅验证所有场景的数据构造和泄漏防护，不执行模型训练。
    for scenario_name, (source_load, target_load) in SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"场景: {scenario_name} ({source_load}HP → {target_load}HP)")
        print(f"{'='*50}")
        
        # 创建场景数据
        try:
            scenario_data = create_scenario(
                data_path=data_path,
                source_load=source_load,
                target_load=target_load,
                window_length=2048,
                source_overlap=0.5,
                target_overlap=0.0,
            )
            
            # 运行泄漏检查
            checker = DataLeakageChecker()
            results = checker.run_all_checks(scenario_data)
            
            if not results['passed']:
                failed_count += 1
                print(f"[FAIL] {scenario_name} 数据泄漏检查失败")
                for check in results['details']['failed']:
                    print(f"   - {check['check']}: {check['detail']}")
                continue
                
            print(f"[PASS] {scenario_name} 数据泄漏检查通过")
            ready_count += 1
            print(f"  源域样本: {scenario_data['source']['num_samples']}")
            print(f"  目标域样本: {scenario_data['target']['num_samples']}")
            
        except FileNotFoundError as e:
            not_ready_count += 1
            print(f"[SKIP] {scenario_name} 原始数据未就绪: {e}")
            continue
        except Exception as e:
            failed_count += 1
            print(f"[FAIL] {scenario_name} 创建失败: {e}")
            continue
            
    print(f"\n{'='*50}")
    print(
        f"场景协议检查完成: {ready_count} READY, "
        f"{failed_count} FAILED, {not_ready_count} NOT READY"
    )
    print("注意: 本脚本没有执行 WDCNN 训练、AdaBN 或准确率评估。")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
