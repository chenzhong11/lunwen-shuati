# WDCNN 论文复现项目

## 复现目标

复现论文：**"A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals"**
- 作者: Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen, Zhujun Zhang
- 期刊: Sensors (MDPI), 2017
- DOI: 10.3390/s17020425
- PMC: PMC5336047

## 实验设计

### Primary Experiment: 跨负载域适应

| 场景 | 源域(训练) | 目标域(测试) |
|------|-----------|-------------|
| S1 | 1HP (Dataset A) | 2HP (Dataset B) |
| S2 | 1HP (Dataset A) | 3HP (Dataset C) |
| S3 | 2HP (Dataset B) | 3HP (Dataset C) |
| S4 | 2HP (Dataset B) | 1HP (Dataset A) |
| S5 | 3HP (Dataset C) | 1HP (Dataset A) |
| S6 | 3HP (Dataset C) | 2HP (Dataset B) |

### 方法对比

| 方法 | 论文报告平均准确率 |
|------|------------------|
| WDCNN (Baseline) | ~90.0% |
| WDCNN + AdaBN | ~95.9% |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python scripts/01_download_data.py

# 3. 预处理数据
python scripts/02_preprocess_data.py

# 4. 运行单个场景
python scripts/03_run_scenario.py --scenario S1 --method wdcnn
python scripts/03_run_scenario.py --scenario S1 --method wdcnn_adabn

# 5. 运行全部场景
python scripts/04_run_all_scenarios.py

# 6. 生成对比报告
python scripts/05_generate_report.py
```

## 项目结构

```
wdcnn-phm-reproduction/
├── config/                # 配置文件（带证据等级标注）
├── data/                  # 数据目录
├── src/                   # 源代码
│   ├── data/             # 数据加载和预处理
│   ├── models/           # WDCNN 模型
│   ├── adaptation/       # AdaBN 域适应
│   ├── training/         # 训练和评估
│   └── utils/            # 工具函数
├── scripts/               # 可执行脚本
├── experiments/           # 实验结果
├── reports/               # 审计报告
└── tests/                 # 单元测试
```

## 证据等级说明

本文档中所有参数标注如下：
- **[Paper-Explicit]**: 论文明确说明
- **[Paper-Inferred]**: 论文间接推断
- **[Assumed]**: 复现假设（论文未说明）
- **[CWRU-Standard]**: CWRU 数据集标准配置

