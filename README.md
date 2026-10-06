# A 股行情数据清洗与因子回测流水线

> 量化数据分析实习中沉淀的数据工程流程：从原始日线行情到无前视偏差的因子数据集，再到含交易成本的绩效评估。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)

---

## 这个项目解决什么问题

因子研究里，**90% 的时间花在数据上，而不是模型上**。典型痛点：

| 痛点 | 后果 | 本项目的处理 |
|---|---|---|
| 停牌导致行数错位 | 指标错算、回测失真 | 统一交易日历对齐 |
| 重复 / 缺失记录 | 分组统计偏差 | 双层去重 + 质量报告 |
| 用了当日收盘价生成信号 | **前视偏差**，回测虚高 | 所有信号 `shift(1)` 滞后 |
| 忽略交易成本 | 高估策略收益 | 双边成本计入 |
| 外部数据对不上 | 不知道错在哪 | 逐项交叉校验 + 差异清单 |

其中**前视偏差**是最隐蔽也最致命的：它能让一个毫无 alpha 的策略在回测里表现优异，直到实盘彻底失效。

---

## 快速开始

```bash
pip install -r requirements.txt

# 演示模式：内置合成数据，无需自备行情
python src/pipeline.py --demo
python src/factor_backtest.py --demo

# 使用自己的行情数据（CSV 或 Parquet）
python src/pipeline.py --source path/to/prices.csv
```

**输入数据格式**（必需列）：

```csv
trade_date,stock_code,open,high,low,close,volume
2024-01-02,600000.SH,10.21,10.35,10.18,10.30,45302100
```

---

## 项目结构

```
src/
├── pipeline.py         # 端到端流水线：读取 → 清洗 → 对齐 → 指标 → 校验
├── factor_backtest.py  # 分层回测与绩效指标
└── output/             # 结果输出
```

---

## 核心设计

### 1. 交易日历对齐

不同股票停牌天数不同，若直接按行计算移动平均，会把「停牌缺行」误当作「时间流逝」。本项目以并集交易日历为基准补齐：

```
股票A:  1  2  3  4  5      股票B:  1  2  4  5
对齐后:  1  2  3  4  5            1  2  3  4  5
         ↑ B 的第3 日为停牌占位
```

实现上用 `merge` 而非 `reindex`，规避不同 pandas 版本对混合类型 MultiIndex 的 dtype 兼容差异。

### 2. 前视偏差防护

```python
ma = close.rolling(20).mean()
out["ma20"] = ma
out["ma20_lag1"] = ma.shift(1)   # ← 信号只用昨日可得的均线
```

回测时信号进一步滞后一期（`_lag1`），确保 T 日建仓用的是 T-1 及之前的信息。

### 3. 指标体系

| 指标 | 说明 | 防偏差处理 |
|---|---|---|
| `ma5 / ma20 / ma60` | 移动平均线 | 提供 `lag1` 版本 |
| `ret_1d / ret_5d / ret_20d` | 多周期收益率 | 天然无前视 |
| `vol_20` | 年化波动率 | 分组滚动后剥离索引层 |
| `rsi_14` | 相对强弱指标 | Wilder 平滑 |
| `ma_ratio_20_60` | 均线相对位置（动量代理） | 回测时 `shift(1)` |

### 4. 数据质量校验

自动输出报告并给出通过 / 不通过结论：

```
股票数量      : 50
记录总数      : 25000
日期区间      : 2015-01-01 ~ 2016-11-30
重复记录      : 0 条（0.0000%）
收盘价缺失    : 50 条（0.2000%）
OHLC 逻辑异常: 0 条
单日涨跌幅>22%: 0 条（需人工复核）
交易日不足    : 0 只（< 200 天）
校验结论      : 通过
```

### 5. 交叉校验

与 Wind 等外部终端数据逐项比对，输出差异明细供回查：

```python
from pipeline import load_prices, cross_validate

mine = load_prices("my_prices.csv")
diff = cross_validate(mine, wind_export)
# diff 含 stock_code / trade_date / close_mine / close_ref / rel_diff
```

---

## 因子回测

采用**分层回测**（quantile backtest）：按因子值五等分分组，逐层统计收益，检验因子是否具备单调性——这是判断因子有效性的标准做法。

```python
from factor_backtest import backtest_momentum, compute_performance

returns, group_stats, perf = backtest_momentum(
    df, signal="ma_ratio_20_60", holding=5, cost_bps=8.0
)
```

绩效指标包含累计收益、年化收益、年化波动、夏普比、最大回撤、胜率。

### 关于演示数据的诚实说明

内置演示数据是**几何布朗运动生成的随机游走**，因此回测中动量因子**不会**呈现显著 alpha。这是刻意的：

> 如果一个随机游走序列能跑出漂亮 alpha，说明回测框架本身有前视偏差。

框架的正确性应通过**在真实数据上是否复现已发表的因子表现**来验证，而非在合成数据上追求好看的数字。请用你自己的行情数据跑一遍。

---

## 技术要点

- **pandas 版本兼容**：`groupby().rolling()` 返回双层索引 Series，直接赋值会触发索引错位与 dtype 报错。本项目统一用 `_roll_by_group()` 剥离分组键，并 `reindex` 回原索引：

  ```python
  s.groupby(group).rolling(w).mean().reset_index(level=0, drop=True).reindex(s.index)
  ```

- **双层去重**：整行去重不足以保证 `(代码, 日期)` 唯一，后者才是不重复索引的前提。
- **无显示环境绘图**：使用 `matplotlib` 的 `Agg` 后端，图标题用英文以规避中文字体缺失警告。

---

## 说明

本项目为个人学习作品，数据清洗流程参考了实习中的工程实践，但：

- **不含任何公司内部数据**，演示数据为程序生成的合成数据
- 回测结果**不构成任何投资建议**
- 真实数据请自行从公开数据源获取，并遵守对应数据源的授权条款

## License

MIT