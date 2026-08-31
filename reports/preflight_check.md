# WDCNN 论文复现 Pre-flight Check

更新时间：2026-08-31

## 总体状态：PASS WITH IMPLEMENTATION GAPS

数据协议、模型骨架和泄漏防护可以进入端到端实验实现/验证阶段；但在真实 CWRU 数据缺失时，不能声称论文复现实验已经完成或已经得到 accuracy。

## 已通过的基础检查

| 检查项 | 状态 | 边界 |
|---|---|---|
| 论文目标和证据等级 | PASS | 目标为 Zhang 等 2017 WDCNN 论文；未说明参数保持为复现假设 |
| CWRU 文件/标签映射 | PASS | `CWRU_FILE_MAP` 与 10 类标签映射已实现 |
| 跨负载场景协议 | PASS | `create_scenario()` 定义 S1-S6，source/target 负载不同 |
| 数据泄漏防护 | PASS | 文件集合、provenance、窗口来源和 AdaBN 标签隔离检查已实现 |
| WDCNN 模型骨架 | PASS | 1D raw input，正式 `input_length=2048`，可做前向传播 |
| AdaBN 模块 | PARTIAL | 模块存在；尚未接入正式训练入口并完成真实结果验证 |
| Trainer/Evaluator | PARTIAL | 通用训练和评估类存在；尚无真实 CWRU 运行证据 |
| 原始数据 | NOT READY | 当前 `data/raw/` 没有 `.mat` 文件 |

## 尚未打通的部分

当前不能声称已经完成：

- 完整 WDCNN + AdaBN 端到端实验；
- S2-S6 自动训练；
- 20 次重复实验、混淆矩阵批量汇总和论文对比报告；
- 独立的 `.mat → .npz` 正式预处理路线。

## 当前数据流

```text
raw .mat
  → create_scenario(source_load, target_load)
  → 在线 2048 分段 + z-score
  → CWRUDataset/DataLoader
  → 1D raw WDCNN
```

`02_preprocess_data.py` 现在只是 raw 数据就绪检查，不是正式必经预处理步骤。旧的 `data/processed/*.npz` 1024 样例保留用于历史/通用模块参考，不作为 Phase 1 输入。

## 当前测试语义

在无真实 `.mat` 时，基础单元测试可以通过，但 S1-S6 真实数据场景测试必须显示 `SKIP`。此状态应表述为“所有已执行测试通过，但存在未执行测试”，不能表述为“所有测试通过”。

本轮实际验证：`9 passed, 1 skipped`；模型结构检查通过；S1-S6 场景检查为 `0 READY, 0 FAILED, 6 NOT READY`。

## 下一步许可边界

只有在 `data/raw/` 完整放入正式 CWRU 文件并通过就绪检查后，才运行：

```powershell
D:/Anaconda3/python.exe scripts/03_run_scenario.py --scenario S1 --method wdcnn
```

该命令当前只形成 S1 WDCNN baseline 闭环；它不会自动代表 AdaBN、S2-S6 或论文全部结果。
