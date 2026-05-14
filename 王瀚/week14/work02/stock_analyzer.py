import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 设置中文字体，防止绘图时中文乱码 (根据系统环境可能需要调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stock(symbol, period='1y'):
    """
    分析股票波动并生成可视化图表及交易建议
    
    参数:
        symbol (str): 股票代码
        period (str): 时间周期 (e.g., '1y', '6mo')
    """
    print(f"正在获取 {symbol} 的股票数据...")
    
    # 1. 获取数据
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    if df.empty:
        print("未获取到数据，请检查股票代码是否正确。")
        return

    # 2. 数据处理与指标计算
    # 计算日波动 (当日最高价 - 当日最低价) / 当日收盘价
    df['Daily_Volatility'] = (df['High'] - df['Low']) / df['Close']
    
    # 计算周波动 (这里使用5日滚动窗口的标准差来模拟短期波动趋势)
    df['Weekly_Volatility'] = df['Close'].pct_change().rolling(window=5).std()
    
    # 计算均线用于趋势判断
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 3. 交易建议逻辑 (基于波动率和均值回归)
    # 逻辑：当周波动率处于历史低位(小于10%分位数)且股价在均线上方 -> 潜在买入(盘整突破)
    # 逻辑：当周波动率处于历史高位(大于90%分位数) -> 潜在卖出(风险过大)
    
    vol_low_threshold = df['Weekly_Volatility'].quantile(0.1)
    vol_high_threshold = df['Weekly_Volatility'].quantile(0.9)
    
    buy_signals = (df['Weekly_Volatility'] < vol_low_threshold) & (df['Close'] > df['MA20'])
    sell_signals = df['Weekly_Volatility'] > vol_high_threshold
    
    # 提取建议日期
    buy_dates = df.index[buy_signals]
    sell_dates = df.index[sell_signals]

    # 4. 可视化绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- 上图：股价走势 ---
    ax1.plot(df.index, df['Close'], label='收盘价 (Close)', color='blue', linewidth=1.5)
    ax1.plot(df.index, df['MA20'], label='20日均线 (MA20)', color='orange', linestyle='--')
    
    # 标记买入点 (绿色三角)
    ax1.scatter(buy_dates, df.loc[buy_dates, 'Close'], marker='^', color='green', label='建议买入点', s=100)
    # 标记卖出点 (红色叉)
    ax1.scatter(sell_dates, df.loc[sell_dates, 'Close'], marker='x', color='red', label='建议卖出点', s=100)
    
    ax1.set_title(f'{symbol} 股票价格走势与交易建议 ({period})', fontsize=16)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 下图：波动率分析 ---
    ax2.plot(df.index, df['Daily_Volatility'], label='日波动率', color='gray', alpha=0.5, linewidth=1)
    ax2.plot(df.index, df['Weekly_Volatility'], label='周波动率 (5日滚动标准差)', color='purple', linewidth=2)
    
    # 绘制阈值参考线
    ax2.axhline(y=vol_low_threshold, color='green', linestyle='--', alpha=0.7, label='低波动阈值 (买入参考)')
    ax2.axhline(y=vol_high_threshold, color='red', linestyle='--', alpha=0.7, label='高波动阈值 (卖出参考)')
    
    ax2.set_title('股票波动率分析', fontsize=14)
    ax2.set_ylabel('波动率', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 格式化X轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

    # 5. 输出文字建议
    print("\n--- 交易时间建议分析 ---")
    if not buy_dates.empty:
        print(f"✅ 最佳买入观察期 (低波动+趋势向上): {buy_dates[-1].strftime('%Y-%m-%d')} 等时间点。")
        print("   理由：市场波动率极低，通常是变盘前兆，结合均线向上，适合布局。")
    else:
        print("⚠️ 当前未发现明显的低波动买入信号。")

    if not sell_dates.empty:
        print(f"🛑 最佳卖出观察期 (极高波动): {sell_dates[-1].strftime('%Y-%m-%d')} 等时间点。")
        print("   理由：市场情绪过热，波动率触及历史高位，风险加剧，建议获利了结。")
    else:
        print("✅ 当前未触及极高波动卖出警戒线。")

if __name__ == "__main__":
    # 示例运行：分析苹果公司股票
    analyze_stock('AAPL', period='6mo')
