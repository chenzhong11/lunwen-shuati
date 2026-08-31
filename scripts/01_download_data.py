"""下载 CWRU 数据集"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='下载 CWRU 轴承数据集')
    parser.add_argument('--output', type=str, default='data/raw', help='保存目录')
    parser.add_argument('--verify', action='store_true', help='验证文件完整性')
    args = parser.parse_args()
    
    save_dir = PROJECT_ROOT / args.output
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"数据将保存到: {save_dir}")
    print("请从 CWRU 官方网站下载数据:")
    print("  https://engineering.case.edu/bearingdatacenter")
    print("\n需要下载的文件:")
    files = ['97.mat', '105.mat', '118.mat', '130.mat', '169.mat', 
             '185.mat', '197.mat', '209.mat', '222.mat', '234.mat']
    for f in files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
