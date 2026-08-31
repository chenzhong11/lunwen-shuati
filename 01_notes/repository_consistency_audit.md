# 仓库一致性审计

更新时间：2026-08-31

## 1. 本轮发现的真实问题

1. 原嵌套仓库的 `reports/preflight_check.md` 将基础协议检查写成“PASS / 可执行实验”，掩盖了训练、评估和结果生成尚未打通。
2. README 的快速开始把 02–05 脚本写成完整复现实验流程，但 02、03、04、05 原先分别只是打印步骤、创建场景、批量检查场景和打印路径。
3. 原 `tests/test_create_scenario.py` 在真实数据缺失时通过 `return None` 表示跳过，手工汇总仍可能输出“所有测试通过”；而 raw 空目录也可能被当作可用数据目录。
4. `test_data_module.py` 使用了已经变化的数据泄漏检查 API，且 CWRU loader 测试读取不存在的 `load_conditions` 属性，导致 pytest 实际失败。
5. `src/models/wdcnn.py` 文件头引用了另一篇论文标题，和仓库目标论文不一致。
6. `1024` 与正式 `2048` 输入长度并存，但原有通用示例没有明确标注用途。
7. 原 `.gitignore` 中未锚定的 `data/` 规则误忽略了 `src/data/`，导致正式数据模块源文件没有被 Git 跟踪。

## 2. 已修复问题

| 问题 | 修复 |
|---|---|
| 状态夸大 | Pre-flight、README、进度报告改为 DONE/PARTIAL/NOT IMPLEMENTED，并明确“PASS WITH IMPLEMENTATION GAPS” |
| 快速开始误导 | 02 改为 raw 就绪检查；03 增加 S1 WDCNN baseline 最小闭环；04 明确只做 S1-S6 场景/泄漏检查；05 实际读取 JSON，无结果时写 NOT READY |
| SKIP 语义 | 使用 pytest 标准 `pytest.skip()`；手工运行汇总区分 PASS/FAIL/SKIP，存在 SKIP 时不再说“所有测试通过” |
| 旧烟测失败 | `test_data_module.py` 对齐当前 `src.data` API，并使用当前泄漏检查接口 |
| 论文来源 | WDCNN 模块文件头统一为当前目标论文；AdaBN 第三方来源仍单独标注为 Li et al. 2016 |
| 忽略规则 | `.gitignore` 改为只忽略 `03_data/`，不再忽略 `02_code/src/data/` |
| 空/部分数据误运行 | `create_scenario()` 对正式请求类别的缺失 `.mat` 文件快速失败 |
| 输入长度 | 正式 Phase 1 统一 2048；1024 仅保留在 legacy/teaching/generic 示例并显式标注 |

## 3. 当前真实完成度

| 能力 | 状态 | 证据/边界 |
|---|---|---|
| 论文理解 | DONE | 目标论文、数据、跨负载协议和证据等级已记录 |
| 数据协议 | DONE | S1-S6 与 source/target 分离已定义 |
| 数据加载 | DONE | `CWRULoader`、`.mat` 通道读取和 `create_scenario()` 已实现 |
| 模型 | DONE | 1D raw WDCNN，默认输入 2048，可前向传播 |
| AdaBN | PARTIAL | 模块实现，但没有正式入口和真实结果 |
| 训练 | PARTIAL | `Trainer` 存在，S1 baseline 入口已接入；尚无真实数据训练证据 |
| 评估 | PARTIAL | `Evaluator` 可输出 accuracy/混淆矩阵；尚无真实数据评估证据 |
| 批量实验 | NOT IMPLEMENTED | 04 当前只做 S1-S6 场景协议和泄漏检查 |
| 结果报告 | PARTIAL | 05 可汇总已完成 JSON；当前没有可汇总的真实实验结果 |
| Day 4 教学线 | DONE | CWT 二维图像 + 2D CNN 教学实验已完成，与 WDCNN 分开 |

## 4. 当前真实可运行命令

```powershell
D:/Anaconda3/python.exe 02_code/src/models/verify_model.py
D:/Anaconda3/python.exe -m pytest -q
D:/Anaconda3/python.exe 02_code/scripts/02_preprocess_data.py
D:/Anaconda3/python.exe 02_code/scripts/04_run_all_scenarios.py --method both
D:/Anaconda3/python.exe 02_code/scripts/05_generate_report.py
```

在完整 CWRU `.mat` 数据准备好后，另可运行：

```powershell
D:/Anaconda3/python.exe 02_code/scripts/03_run_scenario.py --scenario S1 --method wdcnn
```

该命令实际完成 `1HP source → 训练 WDCNN → 2HP target 评估`，结果必须以命令实际写出的 JSON 为准。

## 5. 尚未实现或不能声称完成的命令

- `02_code/scripts/03_run_scenario.py --method wdcnn_adabn`：当前明确返回未实现，不会伪造结果。
- 04 的 `--method` 只是兼容参数，当前不会训练任何方法。
- 20 次重复实验、S2-S6 自动训练、AdaBN 批量对比和论文结果表尚未实现。
- 02 不生成 `.npz`；正式数据流不要求独立 processed 中间层。

## 6. 1024 / 2048 审计结果

| 位置 | 当前值 | 应属用途 | 是否冲突 | 处理 |
|---|---:|---|---|---|
| `02_code/config/default.yaml:data.sample_length` | 2048 | 正式 Phase 1 数据协议 | 否 | 保留为正式标准 |
| `02_code/config/default.yaml:model.input_length` | 2048 | 正式 WDCNN | 否 | 保留为正式标准 |
| `02_code/scripts/03_run_scenario.py`、`04_run_all_scenarios.py` | 2048 | 正式场景入口/协议检查 | 否 | 保留并显式传入 |
| `02_code/src/models/wdcnn.py` | 2048 | 正式模型默认输入 | 否 | 保留 |
| `02_code/config/data_config.yaml` | 1024 | legacy/generic 数据模块配置 | 原表述会冲突 | 已标注，不属于 Phase 1 |
| `01_notes/data_module_guide.md` | 1024 | legacy/teaching API 示例 | 原表述会冲突 | 已标注，不属于 Phase 1 |
| `02_code/examples/data_module_example.py` | 1024 | synthetic generic example | 原表述会冲突 | 已标注，不属于 Phase 1 |
| `03_data/processed/*.npz` | 1024 | 历史本地样例 | 不属于正式协议 | 保留，不上传、不删除 |

## 7. 下一步最小闭环

当前 S1 baseline 的准备条件已经形成，但本机尚未有真实 CWRU 文件。缺口只有：

1. 准备 `03_data/raw/` 中正式映射所需的 CWRU `.mat` 文件；
2. 运行 02 就绪检查和 pytest，确认真实场景测试不再 SKIP；
3. 运行 S1 WDCNN baseline，核验训练、target accuracy、checkpoint 和 JSON 结果；
4. 之后才进入 AdaBN 接入，不扩展到 S2-S6 批量实验。
