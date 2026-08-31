"""根据已完成实验 JSON 生成论文对比报告。

没有实验结果时也会生成明确的 NOT READY 报告，不会填入虚构 accuracy。
"""
import argparse
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='生成论文对比报告')
    parser.add_argument('--results', type=str, default='experiments/results', help='结果目录')
    parser.add_argument('--output', type=str, default='reports/paper_comparison.md', help='输出文件')
    args = parser.parse_args()
    
    results_dir = PROJECT_ROOT / args.results
    output_path = PROJECT_ROOT / args.output
    result_rows = []
    if results_dir.exists():
        for path in sorted(results_dir.rglob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get('status') == 'COMPLETED' and 'accuracy' in payload:
                result_rows.append((path, payload))

    lines = [
        '# 实验结果汇总（自动生成）',
        '',
        '> 本报告只汇总本地已写入的实验 JSON；不会使用论文报告值替代本地结果。',
        '',
    ]
    if not result_rows:
        lines.extend([
            '## 当前状态: NOT READY',
            '',
            f'结果目录 `{args.results}` 中没有可汇总的 COMPLETED 实验结果。',
            '当前不能生成论文对比 accuracy。',
        ])
    else:
        lines.extend([
            '## 当前状态: RESULTS AVAILABLE',
            '',
            '| 场景 | 方法 | Target accuracy | 样本数 | 来源文件 |',
            '|---|---|---:|---:|---|',
        ])
        for path, payload in result_rows:
            lines.append(
                f"| {payload.get('scenario', '未知')} | "
                f"{payload.get('method', '未知')} | "
                f"{payload['accuracy']:.6f} | "
                f"{payload.get('total_samples', '未知')} | `{path.name}` |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'已生成: {output_path}')
    print(f'汇总结果数: {len(result_rows)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
