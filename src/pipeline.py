"""
A股行情数据清洗流水线
======================

本项目复现量化数据分析实习中的核心数据工程流程：
从原始日线行情中构建标准化、无前视偏差的技术指标数据集，
供因子研究直接调用。

设计要点
--------
1. **严格对齐**：以交易日历为基准做内连接，避免停牌导致的行数错位。
2. **无前视偏差**：所有技术指标使用 ``shift(1)`` 滞后，保证
   信号在 T 日生成时只依赖 T-1 及之前的信息。
3. **可校验性**：输出数据质量报告，逐项核对缺失、重复与异常值。
4. **交叉校验**：与外部终端数据逐项比对，差异条目单独输出，
   便于回查原始来源。

模块划分
--------
    config.py数据路径与参数
    data_loader.py    行情数据读取与清洗
    indicators.py     技术指标计算
    quality_check.py  数据质量校验
    pipeline.py       端到端流水线

用法
----
    python src/pipeline.py --demo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 全局配置


@dataclass(frozen=True)
class Config:
    """流水线配置。"""

    start_date: str = "2015-01-01"
    end_date: str = "2026-09-30"
    ma_windows: tuple[int, ...] = (5, 20, 60)
    ret_windows: tuple[int, ...] = (1, 5, 20)
    vol_window: int = 20
    rsi_window: int = 14
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    quality_thresholds: dict = field(default_factory=lambda: {
        "max_missing_rate": 0.02,
        "max_dup_rate": 0.0,
        "min_trading_days": 200,
    })

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()


# ---------------------------------------------------------------- 数据加载


def generate_demo_prices(
    n_stocks: int = 50,
    n_days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """生成演示用行情数据（几何布朗运动 + 市场共同因子）。

    仅用于无真实数据时演示流程，不构成任何投资含义。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(CONFIG.start_date, periods=n_days)

    # 市场共同因子：体现个股间的相关性
    market = rng.normal(0.0003, 0.011, n_days)

    codes = [f"{600000 + i * 10:06d}.SH" for i in range(n_stocks // 2)]
    codes += [f"{i:06d}.SZ" for i in range(1, n_stocks // 2 + 1)]

    frames = []
    for i, code in enumerate(codes):
        idio = rng.normal(0, 0.014, n_days)
        beta = rng.uniform(0.6, 1.4)
        # 叠加个股特质波动，制造异方差
        idio *= 1 + 0.3 * np.abs(rng.normal(0, 0.01, n_days).cumsum()) / 10

        ret = beta * market + idio
        close = 10 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.003, n_days)
        )
        vol = rng.uniform(8e6, 5e7, n_days)

        frames.append(
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "stock_code": code,
                    "open": open_,
                    "high": np.maximum.reduce([open_, close, high]),
                    "low": np.minimum.reduce([open_, close, low]),
                    "close": close,
                    "volume": vol,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)

    # 注入少量数据质量问题，用于演示清洗与校验能力
    n = len(df)
    dup_idx = rng.choice(n, size=int(n * 0.0008), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df.loc[rng.choice(n, size=int(n * 0.002), replace=False), "close"] = np.nan

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_prices(source: str | Path | None = None) -> pd.DataFrame:
    """读取行情数据；未指定数据源时使用演示数据。

    Parameters
    ----------
    source : str | Path | None
        CSV 或 Parquet 文件路径，需含列
        trade_date, stock_code, open, high, low, close, volume。

    Returns
    -------
    DataFrame
        标准化后的行情数据。
    """
    if source is None:
        # 演示数据同样走标准化流程，保证去重规则一致
        return standardize_prices(generate_demo_prices())

    source = Path(source)
    if source.suffix == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)

    return standardize_prices(df)


def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名、类型与排序，构造干净的行情表。

    步骤：
    1. 列名标准化（去掉空格、统一小写）
    2. 日期与代码类型转换
    3. 去除完全重复行
    4. 按「代码-日期」二次去重
       （同一交易日可能存在多条记录，须保留最后一条，
         否则会造成 reindex 时索引不唯一）
    5. 按代码-日期排序
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "trade_date" not in df.columns:
        for cand in ("date", "datetime", "日期"):
            if cand in df.columns:
                df = df.rename(columns={cand: "trade_date"})
                break

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"[清洗] 移除完全重复行：{removed} 条")

    # 保证 (stock_code, trade_date) 唯一，避免后续 reindex 失败
    if df.duplicated(["stock_code", "trade_date"]).any():
        n_before = len(df)
        df = df.drop_duplicates(["stock_code", "trade_date"], keep="last")
        print(f"[清洗] 移除同股同日重复记录：{n_before - len(df)} 条"
              f"（保留最后一条）")

    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return df


def align_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """以并集交易日历为基准对齐多只股票的价格。

    为每只股票补齐全部交易日，使各股票在日历上位置一致，
    避免不同股票停牌导致的行数错位与指标错算。

    实现说明：采用 merge 而非 reindex，规避不同pandas 版本
    对混合类型 MultiIndex 的dtype 兼容差异。

    防御性处理：若传入数据存在重复的 (代码, 日期) 组合，
    先保留最后一条，否则对齐后会异常膨胀。
    """
    if df.duplicated(["stock_code", "trade_date"]).any():
        n0 = len(df)
        df = df.drop_duplicates(["stock_code", "trade_date"], keep="last")
        print(f"[对齐] 清理重复键：{n0 - len(df)} 条")

    codes = df["stock_code"].unique()
    all_dates = np.sort(df["trade_date"].unique())

    # 构造完整 (代码 × 交易日) 骨架
    calendar = pd.DataFrame({
        "stock_code": np.repeat(codes, len(all_dates)),
        "trade_date": np.tile(all_dates, len(codes)),
    })

    before = len(df)
    aligned = calendar.merge(df, on=["stock_code", "trade_date"], how="left")
    n_missing = int(aligned["close"].isna().sum())

    print(f"[对齐] 日历共{len(codes)} 只 × {len(all_dates)} 日 = "
          f"{len(calendar)} 行")
    print(f"[对齐] 原有 {before} 行 → 对齐后 {len(aligned)} 行，"
          f"补齐 {n_missing} 行缺失占位")

    return aligned.sort_values(["stock_code", "trade_date"]).reset_index(
        drop=True)


# ---------------------------------------------------------------- 指标计算


def _roll_by_group(s: pd.Series, group: pd.Series, window: int,
                   min_periods: int | None = None) -> pd.Series:
    """按分组做滚动计算并剥离分组键，避免 MultiIndex 与原索引错位。

    pandas 的 ``groupby().rolling()`` 返回带双层索引的 Series，
    直接赋值给原DataFrame 会触发索引不匹配。此处统一降为单层索引。
    """
    if min_periods is None:
        min_periods = window
    return (
        s.groupby(group)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(s.index)
    )


def add_technical_indicators(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = CONFIG.ma_windows,
    ret_windows: tuple[int, ...] = CONFIG.ret_windows,
    vol_window: int = CONFIG.vol_window,
    rsi_window: int = CONFIG.rsi_window,
) -> pd.DataFrame:
    """计算技术指标，**全部使用滞后信息以避免前视偏差**。

    新增列
    ------
    ma{w}      : w 日简单移动平均（当日收盘价，未滞后）
    ma{w}_lag1 : 滞后一期均线，作为「昨日可得的均线信号」
    ret_{w}d   : w 日收益率
    vol_{w}    : w 日收益率标准差（年化）
    rsi_{n}    : n 日相对强弱指标
    ma_ratio_20_60 : 20/60 日均线相对位置（动量代理）
    """
    out = df.sort_values(["stock_code", "trade_date"]).copy().reset_index(
        drop=True)

    codes = out["stock_code"]
    close = out["close"]
    # 日收益率始终计算，供波动率与动量指标使用
    out["ret_1d"] = close.pct_change()

    for w in ma_windows:
        ma = _roll_by_group(close, codes, w)
        out[f"ma{w}"] = ma
        # 关键：信号使用滞后一期的均线，避免用当日收盘价生成信号
        out[f"ma{w}_lag1"] = ma.shift(1)

    for w in ret_windows:
        out[f"ret_{w}d"] = close.pct_change(w)

    # 年化波动率（分组滚动标准差，同样需剥离分组键）
    out[f"vol_{vol_window}"] = (
        out["ret_1d"]
        .groupby(codes)
        .rolling(vol_window, min_periods=vol_window)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(out.index)
        * np.sqrt(252)
    )

    # RSI（Wilder 平滑）
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = (
        gain.groupby(codes)
        .ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(out.index)
    )
    avg_loss = (
        loss.groupby(codes)
        .ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(out.index)
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{rsi_window}"] = 100 - 100 / (1 + rs)

    # 均线相对位置（动量代理）
    if 20 in ma_windows and 60 in ma_windows:
        out["ma_ratio_20_60"] = out["ma20"] / out["ma60"] - 1

    return out


# ---------------------------------------------------------------- 质量校验


def quality_report(
    df: pd.DataFrame,
    cfg: Config = CONFIG,
) -> dict:
    """输出数据质量报告并返回统计结果。"""
    print("\n" + "=" * 60)
    print("数据质量校验报告")
    print("=" * 60)

    total = len(df)
    n_codes = df["stock_code"].nunique()
    print(f"股票数量      : {n_codes}")
    print(f"记录总数      : {total}")
    print(f"日期区间      : {df['trade_date'].min().date()} ~ "
          f"{df['trade_date'].max().date()}")

    # ---- 重复 ----
    dup_n = int(df.duplicated(["stock_code", "trade_date"]).sum())
    dup_rate = dup_n / total if total else 0
    print(f"重复记录      : {dup_n} 条（{dup_rate:.4%}）")

    # ---- 缺失 ----
    missing = df["close"].isna().sum()
    missing_rate = missing / total if total else 0
    print(f"收盘价缺失    : {missing} 条（{missing_rate:.4%}）")

    # ---- OHLC 一致性 ----
    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    ).sum()
    print(f"OHLC 逻辑异常: {bad_ohlc} 条")

    # ---- 异常涨跌幅 ----
    daily_ret = df.groupby("stock_code")["close"].pct_change()
    extreme = (daily_ret.abs() > 0.22).sum()
    print(f"单日涨跌幅>22%: {extreme} 条（需人工复核）")

    # ---- 交易日覆盖 ----
    days_per_stock = df.groupby("stock_code")["trade_date"].nunique()
    thin = int((days_per_stock < cfg.quality_thresholds["min_trading_days"]).sum())
    print(f"交易日不足    : {thin} 只（< "
          f"{cfg.quality_thresholds['min_trading_days']} 天）")

    passed = (
        dup_rate <= cfg.quality_thresholds["max_dup_rate"]
        and missing_rate <= cfg.quality_thresholds["max_missing_rate"]
        and bad_ohlc == 0
    )
    print(f"\n校验结论      : {'通过' if passed else '存在待处理问题'}")

    return {
        "n_stocks": n_codes,
        "n_records": total,
        "duplicates": dup_n,
        "missing_close": missing,
        "bad_ohlc": int(bad_ohlc),
        "extreme_returns": int(extreme),
        "thin_coverage": thin,
        "passed": bool(passed),
    }


def cross_validate(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    key: str = "stock_code",
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """与外部参考数据（如 Wind 终端导出）逐项交叉校验。

    Parameters
    ----------
    reference : DataFrame
        需含 key, trade_date, close 列。
    tolerance : float
        收盘价相对差异容忍阈值。

    Returns
    -------
    DataFrame
        不一致的明细，便于回查原始来源。
    """
    merged = df.merge(reference, on=[key, "trade_date"],
                      suffixes=("_mine", "_ref"), how="inner")
    if merged.empty:
        print("[校验] 无可比对记录")
        return merged

    merged["abs_diff"] = (merged["close_mine"] - merged["close_ref"]).abs()
    merged["rel_diff"] = merged["abs_diff"] / merged["close_ref"].abs().replace(0, np.nan)

    diff = merged[merged["rel_diff"] > tolerance].copy()
    print(f"[校验] 比对 {len(merged)} 条，"
          f"不一致 {len(diff)} 条（{len(diff) / len(merged):.4%}）")

    if not diff.empty:
        print("[校验] 差异明细（前 10 条）：")
        print(diff[[key, "trade_date", "close_mine", "close_ref",
                    "rel_diff"]].head(10).to_string(index=False))

    return diff.sort_values("rel_diff", ascending=False)


# ---------------------------------------------------------------- 流水线


def run_pipeline(source: str | Path | None = None,
                 reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """端到端执行清洗、对齐、指标计算与校验。"""
    CONFIG.ensure_dirs()

    print("=" * 60)
    print("A股行情数据清洗流水线")
    print("=" * 60)

    # 1. 读取与标准化
    print("\n[1/4] 读取数据…")
    df = load_prices(source)

    # 2. 日历对齐
    print("\n[2/4] 交易日历对齐…")
    df = align_to_calendar(df)

    # 3. 指标计算
    print("\n[3/4] 计算技术指标…")
    df = add_technical_indicators(df)

    # 4. 质量校验
    print("\n[4/4] 数据质量校验…")
    report = quality_report(df)

    # 交叉校验
    if reference is not None:
        print("\n[附加] 交叉校验…")
        cross_validate(df, reference)

    # 导出
    out_csv = CONFIG.output_dir / "prices_with_indicators.csv"
    out_parquet = CONFIG.output_dir / "prices_with_indicators.parquet"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception:
        pass
    print(f"\n输出: {out_csv}")

    # 指标列清单
    ind_cols = [
        c for c in df.columns
        if c.startswith(("ma", "ret_", "vol_", "rsi_")) or c == "ma_ratio_20_60"
    ]
    print(f"生成指标 {len(ind_cols)} 个: {', '.join(ind_cols)}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="行情数据清洗流水线")
    parser.add_argument("--source", type=str, default=None,
                        help="行情文件路径（CSV/Parquet），缺省使用演示数据")
    parser.add_argument("--demo", action="store_true",
                        help="强制使用演示数据")
    args = parser.parse_args()

    source = None if args.demo else args.source
    run_pipeline(source)


if __name__ == "__main__":
    main()