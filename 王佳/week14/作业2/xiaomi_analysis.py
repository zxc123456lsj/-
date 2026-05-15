import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 数据获取 ==========
print("正在获取小米集团(01810.HK)日K线数据...")
df_daily = ak.stock_hk_daily(symbol='01810', adjust='qfq')

# 过滤最近1年数据
cutoff = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily[df_daily['date'] >= cutoff].copy()
df_daily = df_daily.sort_values('date').reset_index(drop=True)

# 构建周K线 (从日线resample)
df_daily_temp = df_daily.set_index('date')
df_weekly = df_daily_temp.resample('W').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna().reset_index()

print(f"日K线数据: {len(df_daily)} 条")
print(f"周K线数据: {len(df_weekly)} 条")

# ========== 2. 技术指标计算 ==========
def compute_indicators(df):
    df = df.copy()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['price_change'] = df['close'].pct_change() * 100
    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

df_daily = compute_indicators(df_daily)
df_weekly = compute_indicators(df_weekly)

# ========== 3. 技术分析 ==========
def identify_trend(df):
    if len(df) < 20:
        return "数据不足"
    latest = df.iloc[-1]
    if pd.notna(latest['ma5']) and pd.notna(latest['ma10']) and pd.notna(latest['ma20']):
        if latest['ma5'] > latest['ma10'] > latest['ma20'] and latest['close'] > latest['ma20']:
            return "上升趋势(多头)"
        if latest['ma5'] < latest['ma10'] < latest['ma20'] and latest['close'] < latest['ma20']:
            return "下降趋势(空头)"
    return "震荡整理"

def find_support_resistance(df, window=20):
    recent = df.tail(window)
    return round(recent['low'].min(), 2), round(recent['high'].max(), 2)

def detect_signals(df):
    signals = []
    for i in range(5, len(df)):
        cur = df.iloc[i]
        prev2 = df.iloc[i-2]
        # 金叉
        if (df.iloc[i-2]['ma5'] <= df.iloc[i-2]['ma10'] and cur['ma5'] > cur['ma10']
            and pd.notna(cur['ma5']) and pd.notna(df.iloc[i-2]['ma5'])):
            signals.append({'date': cur['date'], 'type': 'BUY', 'signal': '金叉(MA5上穿MA10)', 'price': cur['close']})
        # 死叉
        if (df.iloc[i-2]['ma5'] >= df.iloc[i-2]['ma10'] and cur['ma5'] < cur['ma10']
            and pd.notna(cur['ma5']) and pd.notna(df.iloc[i-2]['ma5'])):
            signals.append({'date': cur['date'], 'type': 'SELL', 'signal': '死叉(MA5下穿MA10)', 'price': cur['close']})
        # RSI超卖
        if pd.notna(cur['rsi']) and cur['rsi'] < 30:
            signals.append({'date': cur['date'], 'type': 'BUY', 'signal': 'RSI超卖(<30)', 'price': cur['close']})
        # RSI超买
        if pd.notna(cur['rsi']) and cur['rsi'] > 70:
            signals.append({'date': cur['date'], 'type': 'SELL', 'signal': 'RSI超买(>70)', 'price': cur['close']})
    return signals

def generate_recommendation(week_df, day_df):
    week_trend = identify_trend(week_df)
    day_trend = identify_trend(day_df)
    week_sup, week_res = find_support_resistance(week_df)
    day_sup, day_res = find_support_resistance(day_df)
    week_sigs = detect_signals(week_df)
    day_sigs = detect_signals(day_df)

    latest_close = day_df.iloc[-1]['close']
    latest_rsi = day_df.iloc[-1]['rsi']
    latest_date = day_df.iloc[-1]['date']

    rec = {
        'trend': {'weekly': week_trend, 'daily': day_trend},
        'sr': {'w_support': week_sup, 'w_resistance': week_res,
               'd_support': day_sup, 'd_resistance': day_res},
        'signals': {'weekly': week_sigs[-3:], 'daily': day_sigs[-5:]},
        'current_price': latest_close,
        'current_rsi': latest_rsi,
        'latest_date': latest_date
    }

    # 决策逻辑
    if "上升" in week_trend and "上升" in day_trend:
        if latest_close <= day_sup * 1.03:
            rec['action'] = "买入"
            rec['confidence'] = "高"
            rec['reason'] = "周线与日线共振向上，当前价格接近支撑位，是较好的入场时机"
        else:
            rec['action'] = "持有/加仓"
            rec['confidence'] = "中高"
            rec['reason'] = "双周期多头趋势确认，可继续持有，回调至支撑位附近可加仓"
    elif "下降" in week_trend and "下降" in day_trend:
        rec['action'] = "卖出/观望"
        rec['confidence'] = "高"
        rec['reason'] = "周线与日线共振向下，建议减仓或清仓观望，等待趋势反转信号"
    elif "上升" in week_trend and "震荡" in day_trend:
        rec['action'] = "逢低买入"
        rec['confidence'] = "中"
        rec['reason'] = "周线多头但日线震荡，可在日线支撑位附近分批建仓"
    elif "下降" in week_trend and "震荡" in day_trend:
        rec['action'] = "观望/减仓"
        rec['confidence'] = "中"
        rec['reason'] = "周线空头趋势中，日线震荡反弹可能是下跌中继，建议观望"
    elif "上升" in day_trend and "下降" in week_trend:
        rec['action'] = "短线参与"
        rec['confidence'] = "低"
        rec['reason'] = "日线反弹但周线仍在下降趋势，仅适合短线快进快出"
    else:
        if latest_close <= day_sup * 1.02:
            rec['action'] = "试探性买入"
            rec['confidence'] = "低"
            rec['reason'] = "震荡市中价格接近支撑位，可在支撑位附近小仓位试探"
        else:
            rec['action'] = "观望"
            rec['confidence'] = "低"
            rec['reason'] = "当前处于震荡区间中部，方向不明，建议观望等待突破"

    rec['stop_loss'] = round(day_sup * 0.95, 2)
    rec['target_price'] = round(day_res, 2)
    return rec

rec = generate_recommendation(df_weekly, df_daily)

# ========== 4. 打印分析报告 ==========
print(f"\n{'='*60}")
print(f"  小米集团 (01810.HK) 股票技术分析报告")
print(f"{'='*60}")
print(f" 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f" 最新交易日: {rec['latest_date'].strftime('%Y-%m-%d')}")
print(f" 最新收盘价: HK${rec['current_price']:.2f}")
print(f" 数据区间: {df_daily.iloc[0]['date'].strftime('%Y-%m-%d')} ~ {df_daily.iloc[-1]['date'].strftime('%Y-%m-%d')}")
print(f" 日线数据: {len(df_daily)}条 | 周线数据: {len(df_weekly)}条")

print(f"\n{'─'*60}")
print(f"  [趋势分析]")
print(f"{'─'*60}")
print(f"  周线趋势: {rec['trend']['weekly']}")
print(f"  日线趋势: {rec['trend']['daily']}")
print(f"  RSI(14): {rec['current_rsi']:.1f}")

print(f"\n{'─'*60}")
print(f"  [支撑与压力位]")
print(f"{'─'*60}")
print(f"  周线支撑: HK${rec['sr']['w_support']:.2f}  |  周线压力: HK${rec['sr']['w_resistance']:.2f}")
print(f"  日线支撑: HK${rec['sr']['d_support']:.2f}  |  日线压力: HK${rec['sr']['d_resistance']:.2f}")

print(f"\n{'─'*60}")
print(f"  [操作建议]")
print(f"{'─'*60}")
print(f"  建议动作: {rec['action']}")
print(f"  置信度:   {rec['confidence']}")
print(f"  建议理由: {rec['reason']}")
print(f"  止损位:   HK${rec['stop_loss']:.2f}")
print(f"  目标价位: HK${rec['target_price']:.2f}")

print(f"\n{'─'*60}")
print(f"  [近期技术信号]")
print(f"{'─'*60}")
all_sigs = rec['signals']['daily']
if all_sigs:
    for s in all_sigs:
        d = s['date'].strftime('%Y-%m-%d') if hasattr(s['date'], 'strftime') else str(s['date'])[:10]
        tag = "[买入]" if s['type'] == 'BUY' else "[卖出]"
        print(f"  [{d}] {tag} {s['signal']} @ HK${s['price']:.2f}")
else:
    print(f"  近期无明确技术信号")

# ========== 5. 最近价格走势摘要 ==========
print(f"\n{'─'*60}")
print(f"  [近10个交易日价格变动]")
print(f"{'─'*60}")
recent10 = df_daily.tail(10)
for _, row in recent10.iterrows():
    chg = row['price_change']
    arrow = "+" if chg >= 0 else "-"
    print(f"  {row['date'].strftime('%m-%d')}  开:{row['open']:7.2f}  高:{row['high']:7.2f}  低:{row['low']:7.2f}  收:{row['close']:7.2f}  {arrow}{abs(chg):5.2f}%")

print(f"\n{'='*60}\n")

# ========== 6. 可视化 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 找出日线和周线共同的日期范围
date_min = max(df_daily['date'].min(), df_weekly['date'].min())
date_max = min(df_daily['date'].max(), df_weekly['date'].max())

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(5, 1, height_ratios=[2.5, 0.8, 2.5, 0.8, 0.5], hspace=0.25)

title = "小米集团 (01810.HK) — 周K线与日K线对比分析"
fig.suptitle(title, fontsize=18, fontweight='bold', y=0.99)

# ---- 周K线图 ----
ax1 = fig.add_subplot(gs[0])
ax1.plot(df_weekly['date'], df_weekly['close'], linewidth=2.5, color='#2E86AB', label='周收盘价')
ax1.plot(df_weekly['date'], df_weekly['ma5'], linewidth=1.2, linestyle='--', alpha=0.8, label='MA5')
ax1.plot(df_weekly['date'], df_weekly['ma10'], linewidth=1.2, linestyle='-.', alpha=0.8, label='MA10')
ax1.plot(df_weekly['date'], df_weekly['ma20'], linewidth=1.2, linestyle=':', alpha=0.8, label='MA20')
# 布林带
ax1.fill_between(df_weekly['date'], df_weekly['bb_upper'], df_weekly['bb_lower'],
                 alpha=0.08, color='#2E86AB', label='布林带(±2σ)')
# 标记买卖信号
w_sigs = rec['signals']['weekly']
for s in w_sigs:
    d = s['date']
    color = 'green' if s['type'] == 'BUY' else 'red'
    marker = '^' if s['type'] == 'BUY' else 'v'
    ax1.scatter(d, s['price'], c=color, marker=marker, s=120, zorder=5, edgecolors='black', linewidth=0.5)
ax1.set_ylabel("价格 (港元)", fontsize=11)
ax1.set_title("周K线 + 均线 + 布林带", fontsize=13, fontweight='bold', color='#2E86AB')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')

# ---- 周成交量 ----
ax2 = fig.add_subplot(gs[1], sharex=ax1)
colors_w = ['#00B894' if chg >= 0 else '#D63031' for chg in df_weekly['price_change'].fillna(0)]
ax2.bar(df_weekly['date'], df_weekly['volume'], color=colors_w, alpha=0.6, width=3)
ax2.set_ylabel("成交量", fontsize=10)

# ---- 日K线图 ----
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(df_daily['date'], df_daily['close'], linewidth=1.8, color='#F18F01', label='日收盘价')
ax3.plot(df_daily['date'], df_daily['ma5'], linewidth=1, linestyle='--', alpha=0.7, label='MA5')
ax3.plot(df_daily['date'], df_daily['ma10'], linewidth=1, linestyle='-.', alpha=0.7, label='MA10')
ax3.plot(df_daily['date'], df_daily['ma20'], linewidth=1, linestyle=':', alpha=0.7, label='MA20')
# 布林带
ax3.fill_between(df_daily['date'], df_daily['bb_upper'], df_daily['bb_lower'],
                 alpha=0.06, color='#F18F01')
# 支撑/压力线
ax3.axhline(y=rec['sr']['d_support'], color='green', linestyle='--', linewidth=1, alpha=0.6,
            label=f"支撑: HK${rec['sr']['d_support']:.1f}")
ax3.axhline(y=rec['sr']['d_resistance'], color='red', linestyle='--', linewidth=1, alpha=0.6,
            label=f"压力: HK${rec['sr']['d_resistance']:.1f}")
# 标记买卖信号
d_sigs = rec['signals']['daily']
for s in d_sigs:
    d = s['date']
    color = 'green' if s['type'] == 'BUY' else 'red'
    marker = '^' if s['type'] == 'BUY' else 'v'
    ax3.scatter(d, s['price'], c=color, marker=marker, s=80, zorder=5, edgecolors='black', linewidth=0.5)
ax3.set_ylabel("价格 (港元)", fontsize=11)
ax3.set_title("日K线 + 均线 + 布林带 + 支撑/压力位", fontsize=13, fontweight='bold', color='#F18F01')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3, linestyle='--')

# ---- 日成交量 ----
ax4 = fig.add_subplot(gs[3], sharex=ax1)
colors_d = ['#00B894' if chg >= 0 else '#D63031' for chg in df_daily['price_change'].fillna(0)]
ax4.bar(df_daily['date'], df_daily['volume'], color=colors_d, alpha=0.5, width=0.8)
ax4.set_ylabel("成交量", fontsize=10)

# ---- RSI 指标 ----
ax5 = fig.add_subplot(gs[4], sharex=ax1)
ax5.plot(df_daily['date'], df_daily['rsi'], linewidth=1.2, color='#6C5CE7', label='RSI(14)')
ax5.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
ax5.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
ax5.fill_between(df_daily['date'], 70, df_daily['rsi'].clip(upper=100), alpha=0.15, color='red')
ax5.fill_between(df_daily['date'], 30, df_daily['rsi'].clip(lower=0), alpha=0.15, color='green')
ax5.set_ylabel("RSI", fontsize=10)
ax5.set_ylim(0, 100)
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3, linestyle='--')

# X轴格式化
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 添加注释文本框
textstr = f"建议: {rec['action']} (置信度: {rec['confidence']})\n止损: HK${rec['stop_loss']:.2f} | 目标: HK${rec['target_price']:.2f}"
props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9)
fig.text(0.5, 0.005, textstr, transform=fig.transFigure, fontsize=10,
         verticalalignment='bottom', horizontalalignment='center',
         bbox=props, fontweight='bold')

fig.tight_layout(rect=[0, 0.04, 1, 0.96])

save_path = 'E:/AI学习/课程资料/xiaomi_stock_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图表已保存至: {save_path}")
plt.close()

# ========== 7. 买入时机详细分析 ==========
print(f"\n{'='*60}")
print(f"  [最佳买卖时机深度分析]")
print(f"{'='*60}")

latest_price = rec['current_price']
day_support = rec['sr']['d_support']
day_resistance = rec['sr']['d_resistance']

# 计算价格在支撑-压力区间的位置百分比
sr_range = day_resistance - day_support
if sr_range > 0:
    position_pct = (latest_price - day_support) / sr_range * 100
else:
    position_pct = 50

print(f"  当前价格 HK${latest_price:.2f} 在日线支撑(HK${day_support:.2f})与压力(HK${day_resistance:.2f})之间")
print(f"  距支撑位: {(latest_price - day_support):.2f} ({(latest_price/day_support - 1)*100:+.1f}%)")
print(f"  距压力位: {(day_resistance - latest_price):.2f} ({(day_resistance/latest_price - 1)*100:+.1f}%)")
print(f"  区间位置: {position_pct:.0f}% (0%=支撑位, 100%=压力位)")

print(f"\n  ── 最佳买入时机 ──")
# 基于支撑位计算理想买入区间
buy_zone_low = round(day_support * 0.98, 2)
buy_zone_high = round(day_support * 1.03, 2)
print(f"  理想买入区间: HK${buy_zone_low} ~ HK${buy_zone_high}")
print(f"  触发条件: 价格回踩日线支撑位(HK${day_support:.2f})附近，配合放量企稳或金叉信号")
print(f"  如果价格放量突破日线压力位 HK${day_resistance:.2f} 且周线趋势向好，也是追涨买点")

print(f"\n  ── 最佳卖出时机 ──")
sell_zone = round(day_resistance * 0.97, 2)
print(f"  理想卖出区间: HK${sell_zone} ~ HK${day_resistance:.2f}")
print(f"  触发条件: 价格上涨至压力位附近出现滞涨、RSI超买(>70)或死叉信号")
print(f"  止损卖出: 跌破 HK${rec['stop_loss']:.2f} 应果断止损离场")

# 近期价格走势判断
recent_5 = df_daily.tail(5)
up_days = sum(1 for chg in recent_5['price_change'] if chg > 0)
print(f"\n  ── 近期走势 ──")
print(f"  近5个交易日: {up_days}涨{5-up_days}跌")
print(f"  5日涨幅: {((recent_5.iloc[-1]['close']/recent_5.iloc[0]['close'] - 1)*100):+.2f}%")
print(f"  近5日平均成交量: {recent_5['volume'].mean():.0f}")

# 与历史成交量对比
avg_vol_20 = df_daily.tail(20)['volume'].mean()
last_vol = df_daily.iloc[-1]['volume']
vol_ratio = last_vol / avg_vol_20
print(f"  最新成交量 / 20日均量: {vol_ratio:.2f} ({'放量' if vol_ratio > 1.2 else '缩量' if vol_ratio < 0.8 else '正常'})")

print(f"\n{'='*60}")
print(f"  [免责声明] 以上分析仅基于技术指标，不构成投资建议。")
print(f"     股市有风险，投资需谨慎。请结合基本面、市场情绪等综合判断。")
print(f"{'='*60}\n")
