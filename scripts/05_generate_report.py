"""生成论文对比报告"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='生成论文对比报告')
    parser.add_argument('--results', type=str, default='experiments/results', help='结果目录')
    parser.add_argument('--output', type=str, default='reports/paper_comparison.md', help='输出文件')
    args = parser.parse_args()
    
    print("生成论文对比报告...")
    print(f"读取结果: {args.results}")
    print(f"输出文件: {args.output}")

if __name__ == '__main__':
    main()
