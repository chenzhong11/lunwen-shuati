"""预处理 CWRU 数据"""
import argparse
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='预处理 CWRU 数据')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='配置文件')
    parser.add_argument('--output', type=str, default='data/processed', help='输出目录')
    args = parser.parse_args()
    
    print("数据预处理步骤:")
    print("1. 加载原始 .mat 文件")
    print("2. 按负载条件分离 (1HP/2HP/3HP)")
    print("3. 信号分段 (window=2048, train overlap=50%, test overlap=0%)")
    print("4. z-score 归一化")
    print("5. 保存为 .npz 格式")
    print("\n请确保已下载数据到 data/raw/")

if __name__ == '__main__':
    main()
