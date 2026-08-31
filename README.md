# 论文刷题｜WDCNN 论文复现

本仓库记录 WDCNN 论文的机理学习、数据协议审计和可复现实验实现。当前仓库状态是“基础模块已具备，端到端复现仍在实现中”，不把占位脚本或论文报告值当成本地实验结果。

## 复现目标

当前目标论文：**A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals**

| 字段 | 内容 |
|---|---|
| 作者 | Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen, Zhujun Zhang |
| 期刊/年份 | Sensors (MDPI), 2017 |
| DOI | [10.3390/s17020425](https://doi.org/10.3390/s17020425) |
| PMC | PMC5336047 |
| 正式输入 | 原始一维振动信号，长度 2048 |

## 当前真实状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 论文理解与实验协议 | DONE | 目标论文、CWRU、跨负载 S1-S6、证据等级已记录 |
| 数据加载与场景构造 | DONE | `.mat` 加载、2048 分段、归一化、source/target provenance、泄漏检查 |
| WDCNN 模型 | DONE | 1D WDCNN 结构和前向传播已实现 |
| AdaBN 模块 | PARTIAL | 模块存在，尚未接入正式训练入口 |
| 训练与评估 | PARTIAL | Trainer/Evaluator 和 S1 baseline 入口已具备，尚未使用真实 CWRU 运行 |
| S1 baseline | PARTIAL | 代码闭环已形成；当前因本地 `data/raw/` 无 `.mat` 文件，尚无真实 accuracy |
| S2-S6 自动训练、重复实验、论文对比 | NOT IMPLEMENTED | 不在当前闭环范围内 |
| Day 4 教学线 | DONE | `96×96 CWT + 2D CNN` 仅用于理解卷积，不是 WDCNN 论文输入 |

## 数据流边界

正式 Phase 1 只有一条主数据流：

```text
data/raw/*.mat
  → create_scenario(source_load, target_load)
  → 在线 2048 分段与归一化
  → CWRUDataset / DataLoader
  → 1D raw WDCNN
```

`data/processed/*.npz` 中保留的 1024 样例属于历史/通用数据模块示例，不是正式 Phase 1 输入，也不会上传到 GitHub。`config/data_config.yaml`、`docs/DATA_MODULE_GUIDE.md` 和 `examples/data_module_example.py` 中的 1024 同样已标记为 legacy/teaching only。

## 当前可运行命令

以下命令的描述与实际能力一致：

```powershell
# 模型结构和前向传播检查
D:/Anaconda3/python.exe src/models/verify_model.py

# 单元测试；无真实 .mat 时，S1-S6 场景测试会显示 SKIP
D:/Anaconda3/python.exe -m pytest -q

# 检查 raw 数据是否就绪；不生成 npz
D:/Anaconda3/python.exe scripts/02_preprocess_data.py

# 只检查 S1-S6 场景构造和泄漏防护，不训练
D:/Anaconda3/python.exe scripts/04_run_all_scenarios.py --method both

# 根据已完成的 JSON 结果生成汇总；没有结果时会生成 NOT READY 报告
D:/Anaconda3/python.exe scripts/05_generate_report.py
```

当 `data/raw/` 已放入正式 CWRU `.mat` 文件后，当前唯一已实现的最小训练闭环是：

```powershell
D:/Anaconda3/python.exe scripts/03_run_scenario.py --scenario S1 --method wdcnn
```

它执行 `1HP source → WDCNN 训练 → 2HP target 评估`，并将真实结果写入 `experiments/results/`。当前默认训练配置仍是复现假设，不能直接等同于论文全部未说明参数。

## 尚未实现的正式入口

以下能力不能通过当前脚本声称已经完成：

- WDCNN + AdaBN 端到端评估；
- S2-S6 自动训练与结果汇总；
- 20 次重复实验、混淆矩阵批量比较和论文对比表；
- 从 `.mat` 批量生成正式 `.npz` 的独立预处理路线。

## Day 4 教学线与论文复现线

```text
教学线：96×96 CWT 灰度图 → 2D CNN → 理解卷积、feature map、ReLU、pooling
复现线：2048 点 raw vibration → 1D WDCNN → 论文跨负载实验
```

两条线共享局部感受野、权重共享和逐层组合的卷积思想，但不能把 Day 4 的 CWT 图像教学实验表述为 WDCNN 论文使用 CWT 输入。

## 目录说明

```text
config/       正式配置和场景定义
src/          数据、模型、AdaBN、训练和评估模块
scripts/      数据就绪检查、S1 闭环和状态汇总入口
reports/      pre-flight、进度和一致性审计报告
figures/day4/ Day 4 教学图
tests/        pytest 测试
data/         本地数据目录，不提交原始数据或 processed 样例
```

详细状态见：

- `reports/preflight_check.md`
- `reports/reproduction_progress.md`
- `reports/repository_consistency_audit.md`
- `docs/REPOSITORY_SYNC.md`
