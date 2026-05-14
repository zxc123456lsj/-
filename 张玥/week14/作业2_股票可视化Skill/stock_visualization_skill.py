"""第14周作业 2：股票波动可视化 Skill 实现。

本脚本读取本地股票 CSV，计算日波动和周波动，并绘制到同一张图中。
随后根据波动大小生成教学型买入/卖出建议。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


@dataclass
class VolatilitySummary:
    """股票波动分析摘要。"""

    symbol: str
    start_date: str
    end_date: str
    latest_daily_volatility: float
    latest_weekly_volatility: float
    min_daily_date: str
    min_daily_volatility: float
    max_daily_date: str
    max_daily_volatility: float
    min_weekly_date: str
    min_weekly_volatility: float
    max_weekly_date: str
    max_weekly_volatility: float
    recommendation: str
    chart_path: Path


def load_stock_data(csv_path: Path) -> pd.DataFrame:
    """读取股票 CSV，并完成基础字段校验。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"股票数据文件不存在：{csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = {"date", "close"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV 缺少必要字段：{', '.join(sorted(missing_columns))}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")

    if len(data) < 10:
        raise ValueError("股票数据太少，至少需要 10 条记录才能演示周波动分析。")

    return data.set_index("date")


def calculate_volatility(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算日波动和周波动。

    这里使用涨跌幅绝对值表示波动大小，方便初学者理解。
    """
    daily = data.copy()
    daily["daily_return"] = daily["close"].pct_change()
    daily["daily_volatility"] = daily["daily_return"].abs() * 100
    daily = daily.dropna(subset=["daily_volatility"])

    weekly = data.resample("W-FRI").last().dropna(subset=["close"])
    weekly["weekly_return"] = weekly["close"].pct_change()
    weekly["weekly_volatility"] = weekly["weekly_return"].abs() * 100
    weekly = weekly.dropna(subset=["weekly_volatility"])

    if weekly.empty:
        raise ValueError("周波动数据为空，请提供跨越至少两周的股票数据。")

    return daily, weekly


def make_recommendation(daily: pd.DataFrame, weekly: pd.DataFrame) -> str:
    """根据最近波动和历史极值生成教学型建议。"""
    latest_daily = float(daily["daily_volatility"].iloc[-1])
    latest_weekly = float(weekly["weekly_volatility"].iloc[-1])
    average_daily = float(daily["daily_volatility"].mean())
    average_weekly = float(weekly["weekly_volatility"].mean())
    max_daily_date = daily["daily_volatility"].idxmax().date().isoformat()
    min_daily_date = daily["daily_volatility"].idxmin().date().isoformat()

    if latest_daily > average_daily * 1.5 and latest_weekly > average_weekly * 1.2:
        action = "近期日波动和周波动都明显偏高，短线情绪较激烈，更适合观望或降低仓位。"
    elif latest_daily < average_daily * 0.8 and latest_weekly < average_weekly:
        action = "近期波动低于平均水平，价格节奏相对平稳，可作为分批买入的观察窗口。"
    elif latest_daily > average_daily * 1.5:
        action = "近期日波动突然放大，如果已有持仓，可以考虑设置止盈或止损线。"
    else:
        action = "近期波动处于中等区间，建议结合趋势、成交量和支撑压力位再决策。"

    return (
        f"{action}\n"
        f"历史上日波动最低出现在 {min_daily_date}，可作为相对平稳买点观察；"
        f"日波动最高出现在 {max_daily_date}，可作为风险释放或卖出观察点。"
    )


def plot_volatility(
    symbol: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """把日波动和周波动绘制在同一张图中。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{symbol}_volatility.png"

    # Windows 环境通常有微软雅黑；如果没有，matplotlib 会自动回退到可用字体。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 6))
    plt.plot(
        daily.index,
        daily["daily_volatility"],
        label="日波动",
        color="#2563eb",
        linewidth=1.6,
    )
    plt.plot(
        weekly.index,
        weekly["weekly_volatility"],
        label="周波动",
        color="#dc2626",
        marker="o",
        linewidth=2,
    )
    plt.title(f"{symbol} 日波动与周波动对比")
    plt.xlabel("日期")
    plt.ylabel("波动幅度（%）")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return chart_path


def analyze_stock_volatility(csv_path: Path, symbol: str, output_dir: Path) -> VolatilitySummary:
    """执行完整股票波动分析流程。"""
    data = load_stock_data(csv_path)
    daily, weekly = calculate_volatility(data)
    chart_path = plot_volatility(symbol, daily, weekly, output_dir)
    recommendation = make_recommendation(daily, weekly)

    min_daily_index = daily["daily_volatility"].idxmin()
    max_daily_index = daily["daily_volatility"].idxmax()
    min_weekly_index = weekly["weekly_volatility"].idxmin()
    max_weekly_index = weekly["weekly_volatility"].idxmax()

    return VolatilitySummary(
        symbol=symbol,
        start_date=data.index.min().date().isoformat(),
        end_date=data.index.max().date().isoformat(),
        latest_daily_volatility=float(daily["daily_volatility"].iloc[-1]),
        latest_weekly_volatility=float(weekly["weekly_volatility"].iloc[-1]),
        min_daily_date=min_daily_index.date().isoformat(),
        min_daily_volatility=float(daily.loc[min_daily_index, "daily_volatility"]),
        max_daily_date=max_daily_index.date().isoformat(),
        max_daily_volatility=float(daily.loc[max_daily_index, "daily_volatility"]),
        min_weekly_date=min_weekly_index.date().isoformat(),
        min_weekly_volatility=float(weekly.loc[min_weekly_index, "weekly_volatility"]),
        max_weekly_date=max_weekly_index.date().isoformat(),
        max_weekly_volatility=float(weekly.loc[max_weekly_index, "weekly_volatility"]),
        recommendation=recommendation,
        chart_path=chart_path,
    )


def print_summary(summary: VolatilitySummary) -> None:
    """打印适合作业提交查看的分析结果。"""
    print("=" * 80)
    print(f"股票标识：{summary.symbol}")
    print(f"数据范围：{summary.start_date} 至 {summary.end_date}")
    print(f"最近日波动：{summary.latest_daily_volatility:.2f}%")
    print(f"最近周波动：{summary.latest_weekly_volatility:.2f}%")
    print(
        f"日波动最低：{summary.min_daily_date}，"
        f"{summary.min_daily_volatility:.2f}%"
    )
    print(
        f"日波动最高：{summary.max_daily_date}，"
        f"{summary.max_daily_volatility:.2f}%"
    )
    print(
        f"周波动最低：{summary.min_weekly_date}，"
        f"{summary.min_weekly_volatility:.2f}%"
    )
    print(
        f"周波动最高：{summary.max_weekly_date}，"
        f"{summary.max_weekly_volatility:.2f}%"
    )
    print("\n买入/卖出建议：")
    print(summary.recommendation)
    print(f"\n图表路径：{summary.chart_path}")
    print("提示：以上建议仅用于课程演示，不构成真实投资建议。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="股票日波动和周波动可视化 Skill")
    parser.add_argument("--csv", required=True, help="股票 CSV 文件路径，至少包含 date 和 close 两列")
    parser.add_argument("--symbol", default="DEMO", help="股票标识，用于图表标题和输出文件名")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="图表输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_stock_volatility(
        csv_path=Path(args.csv),
        symbol=args.symbol,
        output_dir=Path(args.output_dir),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
