#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_volatility_chart.py
股票波动率可视化脚本
功能：绘制日波动率与周波动率叠加图 + 买卖信号建议
依赖：pip install akshare matplotlib pandas
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import akshare as ak

# 注册中文字体（Windows 环境）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def fetch_stock_data(stock_code: str, period: str = "daily", adjust: str = "qfq") -> pd.DataFrame:
    """
    通过 akshare 获取股票历史数据
    stock_code: A股如 "000001", 港股如 "00700", 美股如 "AAPL"
    """
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period=period, adjust=adjust)
    except Exception:
        try:
            df = ak.stock_us_spot_em()
            # 兜底：使用正股当日数据接口
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
        except Exception as e:
            print(f"获取数据失败: {e}", file=sys.stderr)
            sys.exit(1)

    df.columns = [c.strip() for c in df.columns]
    # 统一列名
    col_map = {
        '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume',
        '成交额': 'turnover', '涨跌幅': 'pct_chg'
    }
    df.rename(columns=col_map, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算日波动率（每日收益率标准差）与周波动率（5日滚动标准差）
    """
    # 日收益率
    df['daily_return'] = df['close'].pct_change() * 100  # 百分比形式

    # 日波动率（过去5日滚动标准差）
    df['daily_volatility'] = df['daily_return'].rolling(window=5).std()

    # 周波动率（过去20日滚动标准差，模拟月度周期）
    df['weekly_volatility'] = df['daily_return'].rolling(window=20).std()

    # 波动率比值（周/日，>1 说明中期风险加大）
    df['vol_ratio'] = df['weekly_volatility'] / df['daily_volatility']

    return df.dropna()


def generate_signals(df: pd.DataFrame) -> list:
    """
    基于波动率生成买卖信号
    逻辑：
      - 日波动率低位 + 周波动率下降 → 买入信号（震荡筑底）
      - 波动率急剧放大 + 顶背离 → 卖出信号（风险积聚）
      - vol_ratio > 2 极端背离 → 强烈卖出
      - 日波动率 < 过去30日最低 且 周波动率走平 → 低吸窗口
    """
    df = df.copy()
    signals = []

    daily_vol = df['daily_volatility'].values
    weekly_vol = df['weekly_volatility'].values
    vol_ratio = df['vol_ratio'].values
    closes = df['close'].values
    dates = df['date'].values

    for i in range(30, len(df)):
        d_vol = daily_vol[i]
        w_vol = weekly_vol[i]
        ratio = vol_ratio[i]
        close = closes[i]
        date = str(dates[i])[:10]

        signal = None
        strength = 0  # 信号强度 1-5
        reason = ""

        # 条件1: 波动率低位 → 买入
        daily_min_30 = np.min(daily_vol[i-30:i])
        weekly_min_30 = np.min(weekly_vol[i-30:i])

        if d_vol < daily_min_30 * 1.1 and w_vol < weekly_min_30 * 1.15:
            signal = "🟢 买入"
            strength = 3
            reason = f"波动率低位（日:{d_vol:.2f}% 周:{w_vol:.2f}%）"

        # 条件2: 波动率急剧放大（较前日翻倍）→ 卖出
        if daily_vol[i] > daily_vol[i-1] * 2 and ratio > 1.8:
            signal = "🔴 卖出"
            strength = 4
            reason = f"波动率急剧放大（日波放大{ daily_vol[i]/max(daily_vol[i-1],0.01):.1f}倍, 比值{ratio:.1f}）"

        # 条件3: 极端背离 → 强烈卖出
        if ratio > 2.2:
            signal = "🚨 强烈卖出"
            strength = 5
            reason = f"波动率极度背离（比值{ratio:.2f}），风险极高"

        # 条件4: 上涨后波动率未跟随 → 高位预警
        if closes[i] > closes[i-10] * 1.1 and ratio < 1.2 and w_vol > weekly_min_30 * 1.5:
            signal = "🟡 持仓观察"
            strength = 2
            reason = "价格上涨但波动率未跟随，注意回调风险"

        if signal:
            signals.append({
                "date": date,
                "close": round(float(close), 2),
                "signal": signal,
                "strength": strength,
                "reason": reason,
                "daily_volatility": round(float(d_vol), 4),
                "weekly_volatility": round(float(w_vol), 4),
                "vol_ratio": round(float(ratio), 4),
            })

    return signals


def plot_volatility_chart(df: pd.DataFrame, stock_code: str, output_path: str = None):
    """
    绘制波动率叠加图：主图为价格+K线，副图为日/周波动率
    """
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(16, 10),
                                            gridspec_kw={'height_ratios': [2, 1]})

    dates = df['date']

    # ── 主图：价格 + MA ──────────────────────────────────────────────
    ax_price.plot(dates, df['close'], color='#2196F3', linewidth=1.2, label='收盘价')
    ax_price.fill_between(dates, df['close'], alpha=0.15, color='#2196F3')

    # K线颜色（简化版）
    for idx in range(len(df)):
        o = df['open'].iloc[idx]
        c = df['close'].iloc[idx]
        color = '#F44336' if c < o else '#4CAF50'
        ax_price.bar(dates.iloc[idx], height=abs(c - o), bottom=min(o, c),
                      width=0.6, color=color, alpha=0.7)

    ax_price.set_title(f'{stock_code} 价格走势 & 波动率分析', fontsize=16, fontweight='bold')
    ax_price.set_ylabel('价格 (元)', fontsize=12)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    ax_price.set_xlim(dates.min(), dates.max())

    # ── 副图：波动率 ──────────────────────────────────────────────────
    ax_vol.plot(dates, df['daily_volatility'], color='#FF9800', linewidth=1.2,
                label='日波动率 (5日滚动)', linestyle='-')
    ax_vol.plot(dates, df['weekly_volatility'], color='#E91E63', linewidth=1.8,
                label='周波动率 (20日滚动)', linestyle='-')
    ax_vol.fill_between(dates, df['daily_volatility'], alpha=0.2, color='#FF9800')
    ax_vol.fill_between(dates, df['weekly_volatility'], alpha=0.15, color='#E91E63')

    # 买卖信号标注
    signals = generate_signals(df)
    if signals:
        sig_dates = [pd.to_datetime(s['date']) for s in signals]
        sig_prices = [s['close'] for s in signals]
        sig_labels = [s['signal'] for s in signals]

        # 标注在主图
        for sd, sp, sl in zip(sig_dates, sig_prices, sig_labels):
            if '买入' in sl:
                ax_price.annotate(sl, xy=(sd, sp), xytext=(sd, sp * 1.02),
                                  fontsize=8, color='green', fontweight='bold',
                                  arrowprops=dict(arrowstyle='->', color='green'))
            elif '卖出' in sl or '🚨' in sl:
                ax_price.annotate(sl, xy=(sd, sp), xytext=(sd, sp * 0.98),
                                  fontsize=8, color='red', fontweight='bold',
                                  arrowprops=dict(arrowstyle='->', color='red'))

    ax_vol.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='波动率基准线')
    ax_vol.set_title('日波动率 & 周波动率', fontsize=13)
    ax_vol.set_xlabel('日期', fontsize=12)
    ax_vol.set_ylabel('波动率 (%)', fontsize=12)
    ax_vol.legend(loc='upper left')
    ax_vol.grid(True, alpha=0.3)
    ax_vol.set_xlim(dates.min(), dates.max())

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图表已保存: {output_path}")
    else:
        plt.show()

    plt.close()
    return signals


def main():
    parser = argparse.ArgumentParser(description='股票波动率可视化')
    parser.add_argument('--code', '-c', required=True, help='股票代码（A股如 000001）')
    parser.add_argument('--days', '-d', type=int, default=90, help='获取数据天数（默认90）')
    parser.add_argument('--output', '-o', default=None, help='输出图片路径')
    parser.add_argument('--json', '-j', default=None, help='信号JSON输出路径')
    args = parser.parse_args()

    print(f"正在获取 {args.code} 近 {args.days} 天数据...")
    df = fetch_stock_data(args.code, period="daily")
    df = df.tail(args.days).reset_index(drop=True)

    df = calculate_volatility(df)
    signals = plot_volatility_chart(df, args.code, args.output)

    # 信号输出
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(signals, f, ensure_ascii=False, indent=2)
        print(f"信号已保存: {args.json}")

    if signals:
        print(f"\n共生成 {len(signals)} 条交易信号：")
        for s in signals[-10:]:  # 只打印最近10条
            print(f"  {s['date']} | {s['signal']} | {s['reason']}")


if __name__ == '__main__':
    main()
