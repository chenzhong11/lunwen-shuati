# WDCNN 论文复现进度记录

**项目:** WDCNN 轴承故障诊断论文复现  
**记录时间:** 2026-08-21  
**当前状态:** Pre-flight Check 通过，可执行实验

---

## 一、复现目标论文

| 字段 | 内容 |
|------|------|
| **标题** | A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals |
| **作者** | Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen, Zhujun Zhang |
| **期刊** | Sensors (MDPI), 2017 |
| **DOI** | 10.3390/s17020425 |
| **PMC** | PMC5336047 |
| **引用量** | 1704 (截至 2026-08) |
| **Open Access** | ✅ 是 |

---

## 二、Grilling 决策树记录

### Q1-Q4: 基础信息确认

| 问题 | 回答 | 状态 |
|------|------|------|
| Q1 目标论文 | 未确定，需筛选候选 | ✅ 已确定 WDCNN |
| Q2 复现目标 | (a) 验证论文结论 | ✅ |
| Q3 代码/数据 | 无，需调查 | ✅ 已调查 |
| Q4 开发环境 | Win + Anaconda + PyTorch | ✅ |

### Q5: 论文选择

| 选项 | 选择 | 理由 |
|------|------|------|
| A. WDCNN + CWRU | ✅ | Phase 1 首要目标是验证工作流，WDCNN 结构简单 |
| B. WDCNN + CWRU (baseline for few-shot) | - | Phase 2 再考虑 |
| C. Few-shot + MAML + CWRU | - | Phase 2 再考虑 |

### Q6: 论文获取

- 论文 PDF 通过 PMC 全文获取（PMC5336047）
- 确认为 Open Access，可直接访问

### Q7: 未说明参数

| 参数 | 值 | 证据等级 | 来源 |
|------|-----|---------|------|
| Batch Size | 64 | 【复现假设】 | 深度学习通用默认值 |
| 学习率 | 0.001 | 【复现假设】 | Adam 默认值 |
| Epoch 数 | 100 | 【复现假设】 | 经验值 |
| 权重初始化 | He Normal | 【复现假设】 | ReLU+BN 标准 |
| 数据归一化 | z-score | 【复现假设】 | 振动信号标准 |
| 重叠率 | 50% | 【复现假设】 | 数据增强标准 |

### Q8: 数据划分策略

| 选项 | 选择 | 理由 |
|------|------|------|
| (a) 严格复现论文跨负载实验 | ✅ | Phase 1 目标是验证论文结论 |
| (b) 标准随机划分 | - | 不符合论文要求 |

### Q9: AdaBN 实现

| 选项 | 选择 | 说明 |
|------|------|------|
| (b) WDCNN + AdaBN | ✅ | 双轨实验设计 |

**双轨实验设计:**
- Primary Baseline: WDCNN（无 AdaBN）
- Primary Proposed: WDCNN + AdaBN
- 6 种跨负载场景全部执行

### Q10: 实验协议确认

✅ 确认，进入项目结构设计

---

## 三、论文原文核实（12 项参数）

| # | 参数 | 核实结果 | 证据等级 |
|---|------|---------|---------|
| 1 | 样本长度 | 2048 数据点 | 【论文明确】Section 4.1 |
| 2 | 滑动窗口重叠率 | 训练有重叠，测试无重叠 | 【论文明确】重叠存在；【论文未说明】具体值 |
| 3 | Batch Size | 论文未说明 | 【论文未说明】 |
| 4 | 学习率 | 论文未说明 | 【论文未说明】 |
| 5 | 训练 Epoch 数 | 论文未说明 | 【论文未说明】 |
| 6 | 权重初始化 | 论文未说明 | 【论文未说明】 |
| 7 | Batch Normalization | 是，每层使用 | 【论文明确】Section 4.2.2 |
| 8 | Dropout | 不使用 | 【论文明确】Section 3.2 |
| 9 | 数据归一化 | "normalized"，具体方法未说明 | 【论文明确】使用归一化；【论文未说明】具体方法 |
| 10 | 多次运行 | 20 次试验 | 【论文明确】Section 4.3 |
| 11 | 随机种子 | 未固定（通过 20 次试验推断） | 【论文未说明】 |
| 12 | OR 故障位置 | CWRU 默认（6 点钟） | 【论文未说明】 |

---

## 四、WDCNN 架构（论文 Table 2）

| 层 | 类型 | 卷积核/步长 | 通道 | 输出尺寸 | Padding |
|----|------|------------|------|---------|---------|
| 1 | Conv1 | 64×1 / 16×1 | 1→16 | 128×16 | 24 (same) |
| 2 | Pool1 | MaxPool 2×1 / 2×1 | - | 64×16 | - |
| 3 | Conv2 | 3×1 / 1×1 | 16→32 | 64×32 | 1 (same) |
| 4 | Pool2 | MaxPool 2×1 / 2×1 | - | 32×32 | - |
| 5 | Conv3 | 3×1 / 1×1 | 32→64 | 32×64 | 1 (same) |
| 6 | Pool3 | MaxPool 2×1 / 2×1 | - | 16×64 | - |
| 7 | Conv4 | 3×1 / 1×1 | 64→64 | 16×64 | 1 (same) |
| 8 | Pool4 | MaxPool 2×1 / 2×1 | - | 8×64 | - |
| 9 | Conv5 | 3×1 / 1×1 | 64→64 | 6×64 | 0 (valid) |
| 10 | Pool5 | MaxPool 2×1 / 2×1 | - | 3×64 | - |
| 11 | FC1 | - | 192→100 | 100 | - |
| 12 | Output | - | 100→10 | 10 | - |

**总参数:** 54,510  
**激活函数:** ReLU  
**BN:** 每个卷积层和 FC 层后使用

---

## 五、AdaBN 算法（论文 Algorithm 1）

### 算法流程

```
输入: 
  - 目标域无标签信号 {x_t^(1), ..., x_t^(n)}
  - 源域训练得到的 γ, β

输出:
  - 适应后的 WDCNN

算法:
  For 每个 BN 层 i:
    1. 计算目标域均值: μ_t^(i) = E[x_t^(i)]
    2. 计算目标域方差: σ_t^(i)² = Var[x_t^(i)]
    3. 替换 running_mean, running_var
    4. 保持 γ, β 不变
```

### 关键约束

| 约束 | 实现 |
|------|------|
| 目标域数据无标签 | ✅ |
| γ, β 保持源域值 | ✅ |
| 统计量用所有目标样本计算 | ✅ |
| AdaBN 发生在测试前 | ✅ |

---

## 六、跨负载实验设计（论文 Table 3）

| 场景 | 源域(训练) | 目标域(测试) | 论文对比目标 |
|------|-----------|-------------|-------------|
| S1 | 1HP (Dataset A) | 2HP (Dataset B) | Figure 9 |
| S2 | 1HP (Dataset A) | 3HP (Dataset C) | Figure 9 |
| S3 | 2HP (Dataset B) | 3HP (Dataset C) | Figure 9 |
| S4 | 2HP (Dataset B) | 1HP (Dataset A) | Figure 9 |
| S5 | 3HP (Dataset C) | 1HP (Dataset A) | Figure 9 |
| S6 | 3HP (Dataset C) | 2HP (Dataset B) | Figure 9 |

**论文报告结果:**
- WDCNN (Baseline): ~90.0% 平均准确率
- WDCNN + AdaBN: ~95.9% 平均准确率

---

## 七、项目结构

```
E:\projects\wdcnn-phm-reproduction\
├── README.md
├── AGENTS.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── default.yaml          # 主配置（带证据等级标注）
│   ├── scenarios.yaml        # S1-S6 场景定义
│   └── audit/                # 审计文档
├── data/
│   ├── raw/                  # 原始 .mat 文件
│   └── processed/            # 预处理后数据
├── src/
│   ├── data/                 # 数据模块
│   │   ├── cwru_loader.py   # CWRU 加载器 + create_scenario()
│   │   ├── preprocessing.py  # 预处理
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── leakage_checker.py # 泄漏检查
│   ├── models/
│   │   ├── wdcnn.py          # WDCNN 模型
│   │   └── registry.py       # 模型注册表
│   ├── adaptation/
│   │   └── adabn.py          # AdaBN 实现
│   ├── training/
│   │   ├── trainer.py        # 训练器
│   │   └── evaluator.py      # 评估器
│   └── utils/
│       ├── seed.py           # 种子管理
│       ├── logger.py         # 日志
│       ├── metrics.py        # 指标
│       └── env_capture.py    # 环境快照
├── scripts/
│   ├── 01_download_data.py   # 下载数据
│   ├── 02_preprocess_data.py # 预处理
│   ├── 03_run_scenario.py    # 运行单场景
│   ├── 04_run_all_scenarios.py # 运行全部场景
│   └── 05_generate_report.py # 生成报告
├── experiments/
│   ├── results/              # 实验结果
│   ├── figures/              # 图表
│   └── logs/                 # 日志
├── reports/
│   ├── preflight_check.md    # Pre-flight Check 报告
│   └── audit/                # 审计报告
└── tests/
    └── test_create_scenario.py # 单元测试
```

---

## 八、Pre-flight Check 结果

### 总体状态: ✅ PASS

| 类别 | 状态 | 数量 |
|------|------|------|
| PASS | ✅ | 5/7 |
| WARNING | ⚠️ | 2/7 |
| BLOCKER | ❌ | 0/7 (已修复) |

### 已修复的 BLOCKER

**问题:** `get_splits()` 使用随机划分，存在数据泄漏风险

**修复:**
1. `get_splits()` 标记为 legacy，添加 DeprecationWarning
2. 实现 `create_scenario()` 按负载条件分离 source/target
3. 所有 Phase 1 脚本只能通过 `create_scenario()` 获取数据

### 尚存 WARNING

| 项目 | 说明 | 影响 |
|------|------|------|
| CWRU 下载 | 需手动下载或验证 URL | 低 |
| 结果记录 | 需在训练时自动保存 | 中 |

---

## 九、单元测试结果

```
[PASS] 文件映射
[PASS] 标签映射
[PASS] 信号分段
[PASS] z-score 归一化
[PASS] 随机划分警告
[PASS] S1-S6 场景

所有测试通过
```

---

## 十、下一步行动

### 立即执行

1. **下载 CWRU 数据**
   ```bash
   python scripts/01_download_data.py
   ```
   或手动下载到 `data/raw/`

2. **验证数据完整性**
   - 确认 10 个 .mat 文件存在
   - 确认文件大小合理

3. **运行单场景测试**
   ```bash
   python scripts/03_run_scenario.py --scenario S1 --method wdcnn
   ```

### 后续计划

4. **运行全部 6 场景**
   ```bash
   python scripts/04_run_all_scenarios.py --method both
   ```

5. **生成论文对比报告**
   ```bash
   python scripts/05_generate_report.py
   ```

6. **消融实验**（可选）
   - 学习率敏感性
   - Batch Size 敏感性
   - 重叠率敏感性

---

## 十一、关键设计决策汇总

| 决策 | 选择 | 理由 |
|------|------|------|
| 论文选择 | WDCNN (Zhang 2017) | 结构简单，适合验证工作流 |
| 数据集 | CWRU | 论文明确使用，公开可用 |
| 数据划分 | 跨负载评估 | 严格复现论文实验 |
| AdaBN | 实现 | 论文核心贡献 |
| 实验设计 | 双轨（Baseline + AdaBN） | 完整验证论文结论 |
| 框架 | PyTorch | 用户偏好 |
| 随机种子 | 42 | 可重复性 |

---

## 十二、证据等级说明

| 等级 | 含义 |
|------|------|
| 【论文明确】 | 论文原文明确说明 |
| 【论文推断】 | 论文间接推断 |
| 【复现假设】 | 论文未说明，采用默认值 |
| 【第三方实现】 | 来自 GitHub 第三方仓库 |
| 【CWRU 标准】 | CWRU 数据集标准配置 |

---

## 十三、参考文献

1. Zhang, W., et al. (2017). A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals. *Sensors*, 17(2), 425.
2. Li, Y., et al. (2016). Revisiting Batch Normalization for Practical Domain Adaptation. *arXiv:1603.04779*.
3. CWRU Bearing Data Center. https://engineering.case.edu/bearingdatacenter

---

**记录人:** MiMo (AI Assistant)  
**最后更新:** 2026-08-21
