"""检查 CWRU 原始数据是否满足正式 Phase 1 场景要求。

正式数据流不经过本脚本：create_scenario() 直接从 raw .mat 在线分段、
归一化并构造 source/target。保留本脚本作为数据就绪检查工具，避免把
一个没有执行预处理的占位脚本误解成正式必经步骤。
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='检查 CWRU 原始数据就绪状态')
    parser.add_argument('--data-path', type=str, default='data/raw', help='原始 .mat 数据目录')
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data_path
    mat_files = sorted(data_path.glob('*.mat')) if data_path.exists() else []
    print('正式数据流: raw .mat -> create_scenario() -> Dataset/DataLoader')
    print('本脚本只做就绪检查，不生成 processed .npz。')
    print(f'数据目录: {data_path}')
    print(f'.mat 文件数: {len(mat_files)}')
    if not mat_files:
        print('状态: NOT READY（未找到 .mat 文件）')
        return 2
    print('状态: READY（至少发现原始 .mat 文件；完整类别映射由场景命令继续检查）')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
