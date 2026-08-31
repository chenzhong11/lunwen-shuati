# WDCNN 论文复现进度

更新时间：2026-08-31

## 目标论文

| 字段 | 内容 |
|---|---|
| 标题 | A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals |
| 作者 | Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen, Zhujun Zhang |
| 期刊/年份 | Sensors (MDPI), 2017 |
| DOI | 10.3390/s17020425 |
| PMC | PMC5336047 |
| 正式输入 | 原始一维振动信号，长度 2048 |

## 阶段状态

| 阶段 | 状态 | 真实含义 |
|---|---|---|
| 论文理解与复现协议 | DONE | 已确认 CWRU、跨负载 S1-S6、WDCNN 和 AdaBN 目标 |
| 数据加载 | DONE | CWRU `.mat` 加载、信号分段、z-score、provenance 已实现 |
| 泄漏防护 | DONE | source/target 文件不相交、负载不同、provenance 检查已实现 |
| WDCNN | DONE | 1D 模型骨架、2048 输入、前向传播和 BN 层访问已实现 |
| AdaBN | PARTIAL | 统计量适配模块存在，尚未接入正式入口 |
| Trainer/Evaluator | PARTIAL | 训练与评估类存在，真实数据结果尚未产生 |
| S1 WDCNN baseline | PARTIAL | 最小入口已实现；当前缺少本地 CWRU `.mat`，无真实 accuracy |
| S2-S6 自动训练 | NOT IMPLEMENTED | 暂不扩展 |
| 20 次重复与论文对比 | NOT IMPLEMENTED | 暂不扩展 |
| Day 4 CWT-CNN 教学线 | DONE | 仅为 `96×96 CWT + 2D CNN` 机制教学示例 |

## 关键边界

```text
教学线：CWT 二维图像 → 2D CNN
正式复现线：raw 一维振动（2048 点）→ 1D WDCNN
```

Day 4 的图像实验没有修改正式 WDCNN，也不能作为论文使用 CWT 输入的证据。

## 当前阻塞

本地 `data/raw/` 目录存在，但当前没有 `.mat` 文件。因此：

1. `create_scenario()` 会在缺少映射文件时快速失败；
2. pytest 的真实 S1-S6 测试显示 `SKIP`；
3. S1 baseline 代码路径已准备，但尚未实际训练；
4. 任何 accuracy 都不能提前填写或引用论文平均值替代。

本轮验证记录：pytest 为 `9 passed, 1 skipped`；模型前向传播检查通过；S1-S6 在无 raw 数据时均为 `NOT READY`。

## 下一步唯一高优先级任务

准备并核验正式 CWRU 原始文件，然后运行 S1 WDCNN baseline，保存真实训练配置、checkpoint、target accuracy 和混淆矩阵。完成这一闭环后，再决定是否接入 AdaBN。
