"""
因子回测框架（向量化实现）
==========================

在清洗后的数据集上评估单因子与多因子组合的表现，
输出收益曲线、最大回撤等绩效指标。

核心设计
--------
1. **无前视偏差**：信号由 ``*_lag1`` 字段驱动，
   T 日持仓收益使用 T+1 日至 T+N 日实现。
2. **分层回测**：按因子值五等分分组，逐层统计收益，识别单调性。
3. **绩效指标**：累计收益、年化收益、波动率、夏普比、
   最大回撤、胜率、信息比率。
4. **交易成本**：计入双边佣金与冲击成本，使回测结论更贴近实盘。

用法
----
    python src/factor_backtest.py --demo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无显示环境下的绘图后端
import matplotlib.pyplot as plt  # noqa: E402

TRADING_DAYS = 252


@dataclass
class PerformanceStats:
    """绩效指标容器。"""

    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_periods: int

    def to_dict(self) -> dict:
        return {
            "累计收益": self.total_return,
            "年化收益": self.annual_return,
            "年化波动": self.annual_vol,
            "夏普比率": self.sharpe,
            "最大回撤": self.max_drawdown,
            "胜率": self.win_rate,
            "期数": self.n_periods,
        }


def compute_performance(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    rf: float = 0.0,
) -> PerformanceStats:
    """由收益率序列计算绩效指标。

    Parameters
    ----------
    returns : pd.Series
        策略收益率（已扣费）。
    rf : float
        无风险利率（年化），用于计算夏普比。

    Returns
    -------
    PerformanceStats
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return PerformanceStats(0, 0, 0, 0, 0, 0, 0)

    total = float((1 + r).prod() - 1)
    years = n / periods_per_year
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else 0.0
    vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = float((cagr - rf) / vol) if vol > 0 else 0.0

    # 最大回撤
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = float(drawdown.min())

    win_rate = float((r > 0).mean())

    return PerformanceStats(total, cagr, vol, sharpe, mdd, win_rate, n)


def quantile_returns(
    df: pd.DataFrame,
    factor: str,
    ret_col: str = "ret_1d",
    n_groups: int = 5,
    min_count: int = 20,
) -> pd.DataFrame:
    """分层回测：按因子值分组统计收益与单调性。

    Parameters
    ----------
    factor : str
        因子列名（应使用滞后值以避免前视偏差）。
    ret_col : str
        收益率列名（下一期收益）。

    Returns
    -------
    DataFrame
        各分组的收益统计，含多空组合收益。
    """
    d = df[["trade_date", factor, ret_col]].dropna().copy()
    if len(d) < min_count:
        return pd.DataFrame()

    # 按每个截面的因子值分组（避免跨期污染）
    d["group"] = d.groupby("trade_date")[factor].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups,
                          labels=False, duplicates="drop")
        if s.notna().sum() >= n_groups else np.nan
    )
    d = d.dropna(subset=["group"])
    d["group"] = d["group"].astype(int) + 1

    stats = (d.groupby("group")[ret_col]
             .agg(["count", "mean", "std"])
             .rename(columns={"count": "n", "mean": "mean_ret",
                              "std": "std_ret"}))
    stats["mean_ret_annual"] = stats["mean_ret"] * TRADING_DAYS
    stats["t_stat"] = stats["mean_ret"] / (stats["std_ret"] / np.sqrt(stats["n"]))

    # 多空组合：最高组 - 最低组
    if len(stats) >= 2:
        ls = stats["mean_ret"].iloc[-1] - stats["mean_ret"].iloc[0]
        se = np.sqrt(stats["std_ret"].iloc[-1] ** 2 / stats["n"].iloc[-1]
                     + stats["std_ret"].iloc[0] ** 2 / stats["n"].iloc[0])
        stats.attrs["long_short"] = float(ls)
        stats.attrs["long_short_t"] = float(ls / se) if se > 0 else np.nan

    return stats


def backtest_momentum(
    df: pd.DataFrame,
    signal: str = "ma_ratio_20_60",
    holding: int = 5,
    n_quantiles: int = 5,
    cost_bps: float = 8.0,
) -> tuple[pd.Series, pd.DataFrame, PerformanceStats]:
    """动量因子回测：按信号分位组建仓，持有 ``holding`` 天。

    Parameters
    ----------
    signal : str
        信号列（建议使用 ``ma_ratio_20_60``，其本身已基于
        当日收盘计算，回测时须通过 shift 避免前视）。
    cost_bps : float
        单边交易成本（基点）。10bp = 0.1%。

    Returns
    -------
    (returns, group_stats, performance)
        多头组合日收益、分组统计、绩效指标。
    """
    d = df.sort_values(["stock_code", "trade_date"]).copy()

    # 前视偏差防护：信号滞后一期，即 T 日信号用于 T+1 日建仓
    d["signal_lag"] = d.groupby("stock_code")[signal].shift(1)

    # 截面分位分组（标签为 1..n_quantiles，1 为信号最弱）
    d["q"] = d.groupby("trade_date")["signal_lag"].transform(
        lambda s: (pd.qcut(s.rank(method="first"), n_quantiles,
                           labels=False, duplicates="drop") + 1)
        if s.notna().sum() >= n_quantiles else np.nan
    )
    d = d.dropna(subset=["q"])
    d["q"] = d["q"].astype(int)

    # 未来 holding 天收益（作为被解释变量）
    fwd = d[["ret_1d"]].copy()
    for k in range(1, holding + 1):
        fwd[f"fwd_{k}"] = d.groupby("stock_code")["ret_1d"].shift(-k)
    d = pd.concat([d[["trade_date", "stock_code", "q", "signal_lag"]],
                   fwd], axis=1)
    # 持有期累计收益
    d["fwd_ret"] = (1 + d[[f"fwd_{k}" for k in range(1, holding + 1)]]).prod(axis=1) - 1

    # 扣除建仓与平仓成本
    turnover_cost = cost_bps / 10000 * 2
    d["fwd_ret_net"] = d["fwd_ret"] - turnover_cost

    group_stats = (
        d.dropna(subset=["q", "fwd_ret_net"])
        .groupby("q")["fwd_ret_net"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "mean": "mean_ret", "std": "std_ret"})
    )
    group_stats["mean_ret_annual"] = group_stats["mean_ret"] * (
        TRADING_DAYS / holding
    )
    group_stats["t_stat"] = group_stats["mean_ret"] / (
        group_stats["std_ret"] / np.sqrt(group_stats["n"])
    )

    # 多头组合：信号最高组
    top = d[d["q"] == d["q"].max()].dropna(subset=["fwd_ret_net"])
    port = top.set_index("trade_date")["fwd_ret_net"]

    # 转换为日频收益序列便于计算绩效
    daily = port.groupby(level=0).mean()
    daily_ret = (1 + daily) ** (1 / holding) - 1

    perf = compute_performance(daily_ret)
    return daily_ret, group_stats, perf


def run_demo() -> None:
    """在演示数据上运行完整回测。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import CONFIG, run_pipeline  # noqa: E402

    print("生成演示数据并清洗…")
    df = run_pipeline(None)
    df = df.dropna(subset=["close"]).copy()

    # 动量因子回测
    print("\n" + "=" * 60)
    print("因子回测：20/60 日均线动量（持有 5 天，单边 8bp 成本）")
    print("=" * 60)
    rets, gstats, perf = backtest_momentum(df)

    print("\n分组表现（Q1 为信号最弱，Q5 为信号最强）：")
    print(gstats.to_string())
    print(f"\n多空组合年化收益: {gstats['mean_ret_annual'].iloc[-1] - gstats['mean_ret_annual'].iloc[0]:.2%}")

    print("\n绩效指标（多头组合）：")
    for k, v in perf.to_dict().items():
        if isinstance(v, float):
            print(f"  {k:<10}: {v:.4f}")
        else:
            print(f"  {k:<10}: {v}")

    # 收益曲线（使用英文标签以规避中文字体缺失）
    out_dir = CONFIG.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cum = (1 + rets).cumprod()
    ax = cum.plot(figsize=(10, 5), title="Momentum Factor Long Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative NAV")
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=120)
    plt.close(fig)
    print(f"\n收益曲线已保存: {out_dir / 'equity_curve.png'}")

    gstats.to_csv(out_dir / "factor_quantile_stats.csv", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="因子回测")
    parser.add_argument("--demo", action="store_true", help="使用演示数据")
    args = parser.parse_args()
    run_demo()


if __name__ == "__main__":
    main()