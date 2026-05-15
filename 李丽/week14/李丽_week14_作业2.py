import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

def analyze_and_plot(ticker: str):
    print(f"Fetching data for {ticker}...")
    # 使用 akshare 获取近三个月的 A 股数据，免费且无需翻墙/Token
    # 计算最近三个月的起止日期
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=3)
    
    # 移除 ticker 的后缀（如 .SZ 或 .SH），因为 akshare 需要纯数字代码
    clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker
    
    try:
        # 获取 A 股日线数据
        df_daily = ak.stock_zh_a_hist(symbol=clean_ticker, start_date=start_date.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'), adjust="qfq")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    if df_daily is None or df_daily.empty:
        print("No data found.")
        return
        
    # akshare 返回的数据列名为中文，需要适配为代码需要的列名
    df_daily = df_daily.rename(columns={'日期': 'trade_date', '开盘': 'Open', '最高': 'High', '最低': 'Low', '收盘': 'Close'})
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
    df_daily = df_daily.sort_values('trade_date').set_index('trade_date')
    
    # 确保价格列是数值类型
    for col in ['Open', 'High', 'Low', 'Close']:
        df_daily[col] = pd.to_numeric(df_daily[col])
        
    # Calculate daily fluctuation
    df_daily['Daily_Fluctuation'] = (df_daily['High'] - df_daily['Low']) / df_daily['Close'] * 100
    
    # Calculate weekly fluctuation
    df_weekly = df_daily.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'})
    df_weekly['Weekly_Fluctuation'] = (df_weekly['High'] - df_weekly['Low']) / df_weekly['Close'] * 100
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_daily.index, df_daily['Daily_Fluctuation'], label='Daily Fluctuation', color='blue', alpha=0.6)
    ax1.set_ylabel('Daily Fluctuation (%)', color='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(df_weekly.index, df_weekly['Weekly_Fluctuation'], label='Weekly Fluctuation', color='red', marker='o', linestyle='--')
    ax2.set_ylabel('Weekly Fluctuation (%)', color='red')
    
    plt.title(f'{ticker} Fluctuation Analysis')
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    output_file = f'{ticker}_fluctuation.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Recommendation logic
    recent_vol = df_daily['Daily_Fluctuation'].iloc[-1]
    avg_vol = df_daily['Daily_Fluctuation'].mean()
    
    print(f"Recent Volatility: {recent_vol:.2f}%, Average: {avg_vol:.2f}%")
    if recent_vol > avg_vol * 1.5:
        print("Suggestion: SELL - High volatility indicates potential reversal or instability.")
    elif recent_vol < avg_vol * 0.5:
        print("Suggestion: BUY - Low volatility indicates consolidation and potential breakout.")
    else:
        print("Suggestion: HOLD - Fluctuation is within normal ranges.")

if __name__ == "__main__":
    # 使用平安银行的股票代码作为示例
    analyze_and_plot("000001.SZ")
