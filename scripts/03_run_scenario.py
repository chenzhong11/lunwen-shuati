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
from src.data.dataset import CWRUDataset
from src.models import WDCNN
from src.training import Trainer, Evaluator
from src.utils.seed import set_seed

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - dependency is in requirements
    raise SystemExit('运行正式训练需要 PyTorch。') from exc


def main():
    parser = argparse.ArgumentParser(description='运行单个跨负载场景')
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['S1'], help='当前已实现的最小闭环场景')
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

    if args.method == 'wdcnn_adabn':
        print('\n当前脚本尚未实现 AdaBN 端到端入口；本轮仅形成 S1 WDCNN baseline。')
        return 2

    set_seed(args.seed)
    
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
        print("[FAIL] 数据泄漏检查失败，终止实验")
        for check in results['details']['failed']:
            print(f"   - {check['check']}: {check['detail']}")
        return
        
    print("[PASS] 数据泄漏检查通过")
    
    # 输出数据统计
    print(f"\n数据统计:")
    print(f"  源域样本数: {scenario_data['source']['num_samples']}")
    print(f"  目标域样本数: {scenario_data['target']['num_samples']}")
    print(f"  源域文件: {scenario_data['source']['files']}")
    print(f"  目标域文件: {scenario_data['target']['files']}")
    
    source = scenario_data['source']
    target = scenario_data['target']
    if source['num_samples'] == 0 or target['num_samples'] == 0:
        print('\n[FAIL] source 或 target 没有样本，拒绝开始训练')
        return 2

    # 最小闭环：source 有标签训练，target 只在最后用于评估。
    train_dataset = CWRUDataset(signals=source['X'], labels=source['y'])
    target_dataset = CWRUDataset(signals=target['X'], labels=target['y'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=False)

    model = WDCNN(num_classes=10, input_length=2048)
    trainer = Trainer(model, {'lr': 0.001, 'weight_decay': 1e-4})
    trainer.train(train_loader, val_loader=None, epochs=100)

    evaluator = Evaluator(model)
    evaluation = evaluator.evaluate(target_loader)

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f'{args.scenario}_wdcnn_baseline.json'
    checkpoint_path = output_dir / f'{args.scenario}_wdcnn_baseline.pt'
    trainer.save_checkpoint(str(checkpoint_path))
    result = {
        'status': 'COMPLETED',
        'scenario': args.scenario,
        'method': 'wdcnn_baseline',
        'source_load': source_load,
        'target_load': target_load,
        'window_length': 2048,
        'seed': args.seed,
        'epochs': 100,
        'source_samples': source['num_samples'],
        'target_samples': target['num_samples'],
        'accuracy': evaluation['accuracy'],
        'correct_samples': evaluation['correct_samples'],
        'total_samples': evaluation['total_samples'],
        'confusion_matrix': evaluation['confusion_matrix'].tolist(),
        'checkpoint': str(checkpoint_path),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f"\n[PASS] {args.scenario} WDCNN baseline 完成，target accuracy={evaluation['accuracy']:.4f}")
    print(f'结果: {result_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
