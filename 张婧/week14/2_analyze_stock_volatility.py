"""
股票可视化分析 Skill
功能：
1. 获取股票历史数据（支持 A股、美股）
2. 计算日波动（每日振幅百分比）和周波动（每周振幅百分比）
3. 在同一图表中绘制日波动与周波动曲线
4. 基于波动幅度和价格位置给出买入/卖出建议
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import yfinance as yf
from langchain.tools import tool
from typing import Optional, Literal
import warnings
warnings.filterwarnings('ignore')

# 配置 matplotlib 支持中文显示（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@tool
def analyze_stock_volatility(
    symbol: str,
    period: str = "1mo",
    market: Literal["us", "cn"] = "us"
) -> str:
    """
    分析股票的日波动和周波动，绘制波动曲线图，并给出买卖建议。

    Args:
        symbol: 股票代码。美股如 'AAPL'，A股需提供如 '600519.SS'（上交所）或 '000001.SZ'（深交所）
        period: 数据周期，可选 '1mo', '3mo', '6mo', '1y' 等，默认 '1mo'
        market: 市场类型，'us' 表示美股，'cn' 表示 A股。若为 A股，自动添加后缀。

    Returns:
        str: 包含图表(base64编码)和买卖建议的文本描述
    """
    # 处理 A股代码后缀
    if market == "cn":
        if not symbol.endswith(('.SS', '.SZ')):
            if symbol.startswith(('6', '9')):
                symbol = f"{symbol}.SS"
            elif symbol.startswith(('0', '3')):
                symbol = f"{symbol}.SZ"
            else:
                symbol = f"{symbol}.SS"  # 默认上交所

    try:
        # 下载股票数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return f"未获取到股票 {symbol} 的数据，请检查代码或网络。"
        
        # 计算日波动（振幅百分比 = (最高-最低)/收盘 * 100）
        df['daily_volatility'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # 计算周波动：按周重采样，计算每周的（周最高-周最低）/ 周收盘 * 100
        # 先添加周序号
        df['week'] = df.index.isocalendar().week
        df['year'] = df.index.year
        weekly_high = df.groupby(['year', 'week'])['High'].max()
        weekly_low = df.groupby(['year', 'week'])['Low'].min()
        weekly_close = df.groupby(['year', 'week'])['Close'].last()
        weekly_volatility = (weekly_high - weekly_low) / weekly_close * 100
        # 将周波动扩展回日数据（每周同一天）
        week_index = df.groupby(['year', 'week']).apply(lambda x: x.index[0])  # 取每周第一个交易日作为绘图点
        weekly_df = pd.DataFrame({
            'weekly_volatility': weekly_volatility.values
        }, index=week_index)
        df['weekly_volatility'] = weekly_df['weekly_volatility']
        # 向前填充非每周起始日（使每周内相同）
        df['weekly_volatility'] = df['weekly_volatility'].fillna(method='ffill')
        
        # 计算移动平均线（20日）用于趋势判断
        df['ma20'] = df['Close'].rolling(window=20).mean()
        
        # 生成买卖建议
        latest = df.iloc[-1]
        latest_price = latest['Close']
        latest_daily_vol = latest['daily_volatility']
        latest_weekly_vol = latest['weekly_volatility']
        ma20 = latest['ma20'] if not pd.isna(latest['ma20']) else latest_price
        
        # 计算波动率百分位（近20个交易日）
        recent_vol = df['daily_volatility'].tail(20)
        vol_percentile = (recent_vol <= latest_daily_vol).sum() / len(recent_vol) * 100
        
        # 简单策略：
        # - 买入信号：价格低于20日均线 + 当日波动率处于高位（百分位>80） -> 可能超跌反弹
        # - 卖出信号：价格高于20日均线 + 波动率高位 -> 可能高位放量需警惕
        # - 中性：其他情况
        advice = ""
        if latest_price < ma20 and vol_percentile > 80:
            advice = f"🔔 **买入建议**：当前价格 {latest_price:.2f} 低于20日均线 {ma20:.2f}，且波动率处于近20日高位（{vol_percentile:.0f}%分位），可能迎来反弹机会。建议逢低分批买入。"
        elif latest_price > ma20 and vol_percentile > 80:
            advice = f"⚠️ **卖出建议**：当前价格 {latest_price:.2f} 高于20日均线 {ma20:.2f}，波动率显著放大，可能出现回调风险。建议减持或止盈。"
        else:
            advice = f"⚖️ **持有观望**：当前波动率处于中等水平（{vol_percentile:.0f}%分位），价格围绕均线震荡。暂无明确买卖信号。"
        
        # 添加额外信息：周波动趋势
        weekly_trend = "上升" if latest_weekly_vol > df['weekly_volatility'].shift(1).iloc[-1] else "下降"
        advice += f"\n📊 周波动趋势：{weekly_trend}，最新周波动 {latest_weekly_vol:.2f}%。"
        
        # ========== 绘图 ==========
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左侧轴：日波动（蓝色柱状图）
        ax1.bar(df.index, df['daily_volatility'], width=0.8, alpha=0.6, color='steelblue', label='日波动 (%)')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('日波动 (%)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # 右侧轴：周波动（红色折线）
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['weekly_volatility'], color='darkred', linewidth=2, marker='o', markersize=4, label='周波动 (%)')
        ax2.set_ylabel('周波动 (%)', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        # 标题与图例
        title = f'{symbol} 日波动 vs 周波动  ({period})'
        plt.title(title, fontsize=14)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # 将图表转换为 base64 字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # 最终返回内容：文本建议 + 图片 base64
        result = (
            f"股票 {symbol} 波动分析结果（{period}）：\n\n"
            f"最新日期：{df.index[-1].strftime('%Y-%m-%d')}\n"
            f"最新收盘价：{latest_price:.2f}\n"
            f"日波动：{latest_daily_vol:.2f}% | 周波动：{latest_weekly_vol:.2f}%\n"
            f"20日均线：{ma20:.2f}\n\n"
            f"{advice}\n\n"
            f"![波动率图表](data:image/png;base64,{image_base64})"
        )
        return result
    
    except Exception as e:
        return f"分析股票 {symbol} 时出错：{str(e)}。请确保网络通畅且股票代码正确。"


# ========== 使用示例（作为 LangChain Tool 调用） ==========
if __name__ == "__main__":
    # 示例1：分析美股 Apple
    print("=== 测试美股 AAPL ===")
    result = analyze_stock_volatility.invoke({"symbol": "AAPL", "period": "1mo", "market": "us"})
    print(result)
    
    # 示例2：分析 A股贵州茅台（代码 600519）
    print("\n=== 测试 A股 600519 ===")
    result_cn = analyze_stock_volatility.invoke({"symbol": "600519", "period": "3mo", "market": "cn"})
    print(result_cn)
