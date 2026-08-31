# Pre-flight Check Report (更新版)
# WDCNN 论文复现项目
# 生成时间: 2026-08-21
# 更新时间: 2026-08-21 (BLOCKER 修复后)

---

## 总体状态: ✅ PASS (可执行实验)

---

## 1. CWRU 数据下载脚本

### 检查结果: ⚠️ WARNING

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 下载来源 | ✅ | 使用 CWRU 官方 URL |
| 文件映射 | ✅ | 已实现完整的 CWRU_FILE_MAP |
| 文件完整性 | ⚠️ | 无 checksum 验证（建议手动验证） |

**说明:** 下载脚本已实现，但需要手动下载或验证 URL 可用性。

---

## 2. 数据预处理

### 检查结果: ✅ PASS

| 检查项 | 状态 | 论文要求 | 实现 |
|--------|------|---------|------|
| Dataset A/B/C 映射 | ✅ | 1HP/2HP/3HP | CWRU_FILE_MAP 完整 |
| 10 类别映射 | ✅ | Normal + 3×3 faults | FAULT_LABEL_MAP 正确 |
| 窗口长度 | ✅ | 2048 | 2048 |
| 训练集 overlap | ✅ | 50% (推断) | 0.5 |
| 测试集 overlap | ✅ | 0% | 0.0 |
| 归一化 | ✅ | z-score (推断) | z-score |

---

## 3. 数据泄漏检查

### 检查结果: ✅ PASS (BLOCKER 已修复)

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Source/Target 独立性 | ✅ | 来自不同负载的不同 .mat 文件 |
| 文件集合不相交 | ✅ | source_files ∩ target_files == ∅ |
| 负载条件不同 | ✅ | source_load != target_load |
| Provenance 唯一 | ✅ | 无重复 provenance |
| 窗口无重叠 | ✅ | 来自不同文件，无重叠可能 |
| AdaBN 标签隔离 | ✅ | target labels 仅用于评估 |

**修复内容:**
1. 禁用了 `get_splits()` 的随机划分逻辑（标记为 legacy）
2. 实现了 `create_scenario()` 函数，按负载条件分离 source/target
3. 每个样本保存完整 provenance 信息
4. 实现了 DataLeakageChecker 进行自动检查

---

## 4. WDCNN 模型验证

### 检查结果: ✅ PASS

| 层 | Kernel | Stride | Padding | Channels | 参数 |
|----|--------|--------|---------|----------|------|
| Conv1 | 64×1 | 16 | 24 | 1→16 | 1,040 |
| Pool1 | 2×1 | 2 | - | - | 0 |
| Conv2 | 3×1 | 1 | 1 | 16→32 | 1,568 |
| Pool2 | 2×1 | 2 | - | - | 0 |
| Conv3 | 3×1 | 1 | 1 | 32→64 | 6,208 |
| Pool3 | 2×1 | 2 | - | - | 0 |
| Conv4 | 3×1 | 1 | 1 | 64→64 | 12,352 |
| Pool4 | 2×1 | 2 | - | - | 0 |
| Conv5 | 3×1 | 1 | 0 | 64→64 | 12,352 |
| Pool5 | 2×1 | 2 | - | - | 0 |
| FC1 | - | - | - | 192→100 | 19,300 |
| Output | - | - | - | 100→10 | 1,010 |

**总参数: 54,510** ✅

---

## 5. AdaBN 实现验证

### 检查结果: ✅ PASS

| 检查项 | 状态 | 论文要求 | 实现 |
|--------|------|---------|------|
| 仅替换 BN statistics | ✅ | 是 | running_mean, running_var |
| γ/β 保持不变 | ✅ | 是 | 未修改 weight/bias |
| target labels 不参与 | ✅ | 是 | 仅使用 target features |
| 统计量计算方式 | ✅ | 所有 target samples | 拼接所有 batch 后计算 |
| train/eval mode | ✅ | eval mode | compute_target_stats 时切换 |

---

## 6. 实验配置验证

### 检查结果: ✅ PASS

| 场景 | 源域 | 目标域 | 论文要求 | 配置 |
|------|------|--------|---------|------|
| S1 | 1HP | 2HP | 1→2 | ✅ |
| S2 | 1HP | 3HP | 1→3 | ✅ |
| S3 | 2HP | 3HP | 2→3 | ✅ |
| S4 | 2HP | 1HP | 2→1 | ✅ |
| S5 | 3HP | 1HP | 3→1 | ✅ |
| S6 | 3HP | 2HP | 3→2 | ✅ |

---

## 7. 结果记录检查

### 检查结果: ⚠️ WARNING

| 检查项 | 状态 | 说明 |
|--------|------|------|
| config snapshot | ⚠️ | 需在训练时自动保存 |
| git commit | ⚠️ | 需在训练时记录 |
| environment snapshot | ✅ | env_capture.py 已实现 |
| provenance 记录 | ✅ | 每个样本有完整 provenance |

---

## 修改文件列表

### BLOCKER 修复涉及的文件

| 文件 | 修改内容 |
|------|---------|
| `src/data/cwru_loader.py` | 1. 添加 CWRU_FILE_MAP 和 FAULT_LABEL_MAP<br>2. 禁用 get_splits() 随机划分（标记为 legacy）<br>3. 实现 create_scenario() 函数<br>4. 实现 segment_signal() 和 normalize_zscore() |
| `src/data/leakage_checker.py` | 重写为 DataLeakageChecker 类，支持：<br>- 文件集合不相交检查<br>- 负载条件不同检查<br>- Provenance 唯一检查<br>- 窗口重叠检查<br>- AdaBN 标签隔离检查 |
| `src/data/__init__.py` | 更新导出列表 |
| `scripts/03_run_scenario.py` | 使用 create_scenario() 创建场景 |
| `scripts/04_run_all_scenarios.py` | 使用 create_scenario() 创建场景 |
| `tests/test_create_scenario.py` | 新增单元测试 |

### 修改前的问题

- `get_splits()` 使用随机 70/15/15 划分，不同负载的数据被混合
- 同一原始信号的窗口可能同时进入 source 和 target
- 无法保证 source 和 target 来自不同的原始文件

### 修改后的数据流

```
原始 .mat 文件（按负载分离）
    │
    ├─ 1HP 文件 (98.mat, 119.mat, ...)
    ├─ 2HP 文件 (99.mat, 120.mat, ...)
    └─ 3HP 文件 (100.mat, 121.mat, ...)
        │
        ▼
create_scenario(source_load, target_load)
    │
    ├─ source: 仅加载 source_load 对应的文件
    │   └─ 记录 provenance (source_file, load, fault_type, window_start, end)
    │
    └─ target: 仅加载 target_load 对应的文件
        └─ 记录 provenance (source_file, load, fault_type, window_start, end)
        │
        ▼
DataLeakageChecker 验证
    ├─ source_files ∩ target_files == ∅ ✓
    ├─ source_load != target_load ✓
    └─ provenance 无重复 ✓
```

---

## 测试结果

### 单元测试结果

```
[PASS] 文件映射
[PASS] 标签映射
[PASS] 信号分段
[PASS] z-score 归一化
[PASS] 随机划分警告
[PASS] S1-S6 场景

所有测试通过
```

### S1-S6 泄漏检查结果

| 场景 | Source Load | Target Load | Source Files | Target Files | 交集 | 状态 |
|------|-------------|-------------|--------------|--------------|------|------|
| S1 | 1HP | 2HP | 98,119,186,... | 99,120,187,... | ∅ | ✅ |
| S2 | 1HP | 3HP | 98,119,186,... | 100,121,188,... | ∅ | ✅ |
| S3 | 2HP | 3HP | 99,120,187,... | 100,121,188,... | ∅ | ✅ |
| S4 | 2HP | 1HP | 99,120,187,... | 98,119,186,... | ∅ | ✅ |
| S5 | 3HP | 1HP | 100,121,188,... | 98,119,186,... | ∅ | ✅ |
| S6 | 3HP | 2HP | 100,121,188,... | 99,120,187,... | ∅ | ✅ |

---

## 执行许可

**当前状态: ✅ 允许执行实验**

**条件:** 需先下载 CWRU 数据到 `data/raw/` 目录

**下一步:** `python scripts/01_download_data.py`

---

## 尚存 WARNING

| 项目 | 说明 | 影响 |
|------|------|------|
| CWRU 下载 | 需手动下载或验证 URL | 低 |
| 文件完整性 | 无 checksum 验证 | 低 |
| 结果记录 | 需在训练时自动保存 | 中 |
