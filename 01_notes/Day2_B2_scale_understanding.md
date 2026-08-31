# Day2_B2_scale_understanding.md

## 一、实验目的

直观看懂：scale不是图片尺寸，而是CWT使用的小波尺度数量；不同scale对应不同频率响应。

## 二、实验设置

- 采样频率：1000 Hz
- 信号时长：1 秒
- 三个信号：
  - A：100Hz正弦
  - B：200Hz正弦
  - C：100Hz + 200Hz正弦

## 三、核心发现

### 1. scale是什么？

**scale是小波的"拉伸程度"**：
- scale=1：小波被压缩，适合检测高频
- scale=128：小波被拉伸，适合检测低频

**scale不是图片尺寸**：
- 图片尺寸是resize后的结果
- scale是CWT使用的小波尺度数量

### 2. scale为什么和frequency有关？

**反比关系**：
```
frequency ∝ 1/scale
```

**实验验证**：
| Scale | Frequency (Hz) |
|-------|----------------|
| 1 | 812.50 |
| 2 | 406.25 |
| 4 | **203.12** ← 最接近200Hz |
| 8 | **101.56** ← 最接近100Hz |
| 128 | 6.35 |

**物理解释**：
- 小scale的小波振荡快 → 匹配高频信号
- 大scale的小波振荡慢 → 匹配低频信号

### 3. scale增加到底改变了什么？

**改变的是频率分辨率**：

| Scale范围 | 频率范围 | 覆盖100Hz | 覆盖200Hz |
|-----------|----------|-----------|-----------|
| 1~32 | 25.4 ~ 812.5 Hz | ✓ | ✓ |
| 1~64 | 12.7 ~ 812.5 Hz | ✓ | ✓ |
| 1~128 | 6.3 ~ 812.5 Hz | ✓ | ✓ |

**结论**：
- 三种scale范围都能覆盖100Hz和200Hz
- 但scale越大，低频分辨率越高
- scale=128能检测到6.35Hz的低频

### 4. coefficients的两个维度分别是什么？

**coefficients.shape = (128, 1000)**

| 维度 | 含义 |
|------|------|
| 第一维 (axis=0) | **scale维度**：128个不同的小波尺度 |
| 第二维 (axis=1) | **time维度**：1000个时间点 |

**具体元素解释**：

```python
coefficients[0, 0] = 0.1917
```
- 位置：scale索引=0, 时间索引=0
- 含义：第1个scale (scale=1) 在第1个时间点 (t=0s) 的小波系数
- 对应频率：812.50 Hz
- 物理意义：信号在t=0时刻与scale=1的小波的匹配程度

```python
coefficients[20, 500] = -0.0065
```
- 位置：scale索引=20, 时间索引=500
- 含义：第21个scale (scale=21) 在第501个时间点 (t=0.5s) 的小波系数
- 对应频率：38.69 Hz
- 物理意义：信号在t=0.5s时刻与scale=21的小波的匹配程度

```python
coefficients[80, 500] = -1.4535
```
- 位置：scale索引=80, 时间索引=500
- 含义：第81个scale (scale=81) 在第501个时间点 (t=0.5s) 的小波系数
- 对应频率：10.03 Hz
- 物理意义：信号在t=0.5s时刻与scale=81的小波的匹配程度

### 5. CWT图到底是怎么从coefficients矩阵变出来的？

**转换过程**：
```
coefficients矩阵 (128, 1000)
    ↓
取绝对值：np.abs(coefficients)
    ↓
用imshow显示：
  - 横轴：时间点 (0~1000 → 0~1秒)
  - 纵轴：scale (1~128)
  - 颜色：系数幅值
    ↓
得到CWT Scalogram
```

**关键点**：
- CWT图就是coefficients矩阵的可视化
- 颜色越亮，表示该scale在该时间点的系数越大
- 系数大 = 信号与该scale的小波匹配程度高

## 四、三个信号的CWT对比

### 信号A：100Hz正弦
- CWT图上：在scale=8附近（对应100Hz）有持续的高能量
- 其他scale：能量很低

### 信号B：200Hz正弦
- CWT图上：在scale=4附近（对应200Hz）有持续的高能量
- 其他scale：能量很低

### 信号C：100Hz + 200Hz
- CWT图上：在scale=4和scale=8附近都有高能量
- 两个频率成分都被检测到

## 五、关键结论

1. **scale是小波的拉伸程度**，不是图片尺寸
2. **scale与frequency成反比**：f ∝ 1/scale
3. **scale增加**：低频分辨率提高，能检测更低频率
4. **coefficients矩阵**：(n_scales, n_time_points)
5. **CWT图**：coefficients矩阵的可视化，颜色表示匹配程度

## 六、图表文件

- `day2_scale_three_signals.png`：三个信号的CWT对比
- `day2_scale_matrix_explain.png`：矩阵结构可视化

---

**实验完成**
