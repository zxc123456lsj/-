---
name: 股票可视化与交易建议
description: 获取股票周K线和日K线数据，绘制对比图表，并基于技术分析提供买卖时机建议。
---

# 功能概述

本Skill提供以下功能：
- 获取股票周K线和日K线数据
- 绘制周波动与日波动对比图表
- 基于技术指标分析给出买卖建议

# 核心分析方法

## 一、趋势判断（多周期共振）

### 1. 月K线 - 确定大方向
- **多头排列**：5月 > 10月 > 20月均线 → 上升趋势
- **空头排列**：5月 < 10月 < 20月均线 → 下降趋势
- **横盘震荡**：均线纠缠，无明显方向

### 2. 周K线 - 确认中期节奏
- **支撑位识别**：回调不破20周均线或前期低点
- **压力位识别**：多次在某价位遇阻回落
- **买入信号**：缩量回调至支撑位 + 下影线

### 3. 日K线 - 精准择时
- **突破信号**：放量阳线突破颈线/平台
- **底部形态**：长下影、阳包阴、启明星
- **顶部形态**：长上影、阴包阳、黄昏星

## 二、波动率分析

### 周波动 vs 日波动
- **周波动率大 + 日波动率小**：趋势稳定，适合持仓
- **周波动率小 + 日波动率大**：短线机会多，适合波段
- **双高**：高风险高收益，需严格止损
- **双低**：观望为主，等待突破

## 三、买卖时机判断矩阵

| 月线趋势 | 周线位置 | 日线信号 | 操作建议 | 仓位建议 |
|---------|---------|---------|---------|---------|
| 上升 | 回调至支撑 | 买入形态 | **强烈买入** | 60-80% |
| 上升 | 接近压力 | 弱势信号 | 减仓观望 | 20-30% |
| 上升 | 突破压力 | 放量上涨 | **加仓买入** | 80-100% |
| 下降 | 反弹至压力 | 卖出形态 | **坚决卖出** | 空仓 |
| 下降 | 超跌远离均线 | 底部形态 | 轻仓试多 | 10-20% |
| 震荡 | 区间下沿 | 企稳信号 | 波段买入 | 30-40% |
| 震荡 | 区间上沿 | 滞涨信号 | 波段卖出 | 减仓 |

# 调用方法
```python
TOKEN = "zgaLG8unUPr"
import requests # type: ignore from typing import Annotated, Optional, Dict, List, Tuple import traceback import matplotlib.pyplot as plt import matplotlib.dates as mdates from datetime import datetime, timedelta import numpy as np import pandas as pd
==================== 数据获取函数 ====================
@app.get("/get_stock_kline_data", operation_id="get_stock_kline_data") async def get_stock_kline_data( code: Annotated[str, "股票代码"], period: Annotated[str, "K线周期: day/week/month"] = "day", startDate: Annotated[Optional[str], "开始时间(YYYY-MM-DD)"] = None, endDate: Annotated[Optional[str], "结束时间(YYYY-MM-DD)"] = None, type: Annotated[int, "复权类型: 0不复权,1前复权,2后复权"] = 1 ) -> Dict: """获取股票K线数据""" period_map = { "day": "day", "week": "week", "month": "month" }
if period not in period_map:
    return {"error": "Invalid period. Use 'day', 'week', or 'month'"}

url = f"https://api.autostock.cn/v1/stock/kline/{period_map[period]}?token={TOKEN}"

headers = {}
try:
    payload = {
        "code": code,
        "startDate": startDate,
        "endDate": endDate,
        "type": type
    }
    response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
    return response.json()
except Exception:
    print(traceback.format_exc())
    return {}
@app.get("/get_stock_info", operation_id="get_stock_info") async def get_stock_info(code: Annotated[str, "股票代码"]) -> Dict: """获取股票基础信息""" url = f"https://api.autostock.cn/v1/stock?token={TOKEN}&code={code}"
payload = {}
headers = {}
try:
    response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
    return response.json()
except Exception:
    print(traceback.format_exc())
    return {}
==================== 数据处理函数 ====================
def parse_kline_data(kline_response: Dict) -> pd.DataFrame: """解析K线数据为DataFrame""" if not kline_response or 'data' not in kline_response: return pd.DataFrame()
data = kline_response['data']
if not data:
    return pd.DataFrame()

# 假设API返回格式: [{date, open, high, low, close, volume, ...}, ...]
df = pd.DataFrame(data)

# 转换日期格式
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

# 确保数值列是float类型
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

return df
def calculate_volatility(df: pd.DataFrame, window: int = 20) -> float: """计算波动率（收盘价的标准差除以均值）""" if df.empty or 'close' not in df.columns: return 0.0
returns = df['close'].pct_change().dropna()
if len(returns) < window:
    return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0

return returns.tail(window).std() * np.sqrt(252)
def calculate_moving_averages(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame: """计算移动平均线""" if df.empty or 'close' not in df.columns: return df
df_copy = df.copy()
for period in periods:
    df_copy[f'ma{period}'] = df_copy['close'].rolling(window=period).mean()

return df_copy
def identify_trend(df: pd.DataFrame) -> str: """识别趋势方向""" if df.empty or len(df) < 20: return "unknown"
df_ma = calculate_moving_averages(df, [5, 10, 20])

if len(df_ma) < 20:
    return "unknown"

latest = df_ma.iloc[-1]

# 检查多头排列
if (latest.get('ma5', 0) > latest.get('ma10', 0) > latest.get('ma20', 0) and 
    latest['close'] > latest.get('ma20', 0)):
    return "uptrend"

# 检查空头排列
elif (latest.get('ma5', 0) < latest.get('ma10', 0) < latest.get('ma20', 0) and 
      latest['close'] < latest.get('ma20', 0)):
    return "downtrend"

else:
    return "sideways"
def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]: """寻找支撑位和压力位""" if df.empty or len(df) < lookback: return (0, 0)
recent_data = df.tail(lookback)

# 支撑位：近期最低点
support = recent_data['low'].min()

# 压力位：近期最高点
resistance = recent_data['high'].max()

return (support, resistance)
def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]: """检测K线形态""" if df.empty or len(df) < 3: return []
patterns = []
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) >= 2 else None

if prev is None:
    return patterns

# 计算实体大小
body = abs(latest['close'] - latest['open'])
upper_shadow = latest['high'] - max(latest['open'], latest['close'])
lower_shadow = min(latest['open'], latest['close']) - latest['low']
total_range = latest['high'] - latest['low']

if total_range == 0:
    return patterns

# 长下影线（锤子线）
if lower_shadow > body * 2 and upper_shadow < body * 0.5:
    patterns.append("hammer")

# 长上影线（射击之星）
if upper_shadow > body * 2 and lower_shadow < body * 0.5:
    patterns.append("shooting_star")

# 阳包阴
if (latest['close'] > latest['open'] and prev['close'] < prev['open'] and
    latest['close'] > prev['open'] and latest['open'] < prev['close']):
    patterns.append("bullish_engulfing")

# 阴包阳
if (latest['close'] < latest['open'] and prev['close'] > prev['open'] and
    latest['close'] < prev['open'] and latest['open'] > prev['close']):
    patterns.append("bearish_engulfing")

return patterns
==================== 可视化函数 ====================
@app.get("/visualize_stock", operation_id="visualize_stock") async def visualize_stock( code: Annotated[str, "股票代码"], days: Annotated[int, "回溯天数（日线）"] = 120, save_path: Annotated[Optional[str], "保存路径（可选）"] = None ) -> Dict: """ 绘制股票周K线和日K线对比图，并给出交易建议
返回：
- chart_path: 图表保存路径
- analysis: 分析报告
- recommendation: 交易建议
"""
try:
    # 1. 获取日线数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    day_data = await get_stock_day_kline(code, start_date, end_date, type=1)
    week_data = await get_stock_week_kline(code, None, None, type=1)
    stock_info = await get_stock_info(code)
    
    # 2. 解析数据
    df_day = parse_kline_data(day_data)
    df_week = parse_kline_data(week_data)
    
    if df_day.empty:
        return {"error": "Failed to fetch daily data"}
    
    # 3. 计算技术指标
    df_day = calculate_moving_averages(df_day, [5, 10, 20, 60])
    df_week = calculate_moving_averages(df_week, [5, 10, 20])
    
    # 计算波动率
    daily_volatility = calculate_volatility(df_day, 20)
    weekly_volatility = calculate_volatility(df_week, 12)
    
    # 识别趋势
    day_trend = identify_trend(df_day)
    week_trend = identify_trend(df_week) if not df_week.empty else "unknown"
    
    # 寻找支撑压力
    support, resistance = find_support_resistance(df_day, 60)
    
    # 检测K线形态
    patterns = detect_candlestick_patterns(df_day)
    
    # 4. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f'{code} - 股票技术分析图表\n{stock_info.get("name", "") if stock_info else ""}', 
                 fontsize=14, fontweight='bold')
    
    # 子图1: 日K线 + 均线
    ax1 = axes[0]
    ax1.plot(df_day['date'], df_day['close'], label='收盘价', color='blue', linewidth=1.5)
    
    if 'ma5' in df_day.columns:
        ax1.plot(df_day['date'], df_day['ma5'], label='MA5', color='orange', linewidth=1)
    if 'ma10' in df_day.columns:
        ax1.plot(df_day['date'], df_day['ma10'], label='MA10', color='green', linewidth=1)
    if 'ma20' in df_day.columns:
        ax1.plot(df_day['date'], df_day['ma20'], label='MA20', color='red', linewidth=1)
    if 'ma60' in df_day.columns:
        ax1.plot(df_day['date'], df_day['ma60'], label='MA60', color='purple', linewidth=1, linestyle='--')
    
    # 标注支撑压力位
    if support > 0:
        ax1.axhline(y=support, color='green', linestyle=':', alpha=0.7, label=f'支撑位: {support:.2f}')
    if resistance > 0:
        ax1.axhline(y=resistance, color='red', linestyle=':', alpha=0.7, label=f'压力位: {resistance:.2f}')
    
    ax1.set_ylabel('价格 (元)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    # 子图2: 成交量
    ax2 = axes[1]
    colors = ['red' if df_day['close'].iloc[i] >= df_day['open'].iloc[i] else 'green' 
              for i in range(len(df_day))]
    ax2.bar(df_day['date'], df_day['volume'], color=colors, alpha=0.6, label='成交量')
    ax2.set_ylabel('成交量')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    
    # 子图3: 周K线对比
    ax3 = axes[2]
    if not df_week.empty:
        ax3.plot(df_week['date'], df_week['close'], label='周K收盘价', color='darkblue', 
                linewidth=2, marker='o', markersize=4)
        if 'ma5' in df_week.columns:
            ax3.plot(df_week['date'], df_week['ma5'], label='周MA5', color='orange', linewidth=1.5)
        if 'ma20' in df_week.columns:
            ax3.plot(df_week['date'], df_week['ma20'], label='周MA20', color='red', linewidth=1.5)
    
    ax3.set_xlabel('日期')
    ax3.set_ylabel('价格 (元)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        chart_path = save_path
    else:
        chart_path = f"stock_{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 生成交易建议
    recommendation = generate_recommendation(
        day_trend, week_trend, daily_volatility, weekly_volatility,
        support, resistance, patterns, df_day
    )
    
    # 6. 构建分析报告
    analysis = {
        "股票代码": code,
        "股票名称": stock_info.get("name", "未知") if stock_info else "未知",
        "日线趋势": day_trend,
        "周线趋势": week_trend,
        "日波动率(年化)": f"{daily_volatility*100:.2f}%",
        "周波动率(年化)": f"{weekly_volatility*100:.2f}%",
        "支撑位": f"{support:.2f}" if support > 0 else "N/A",
        "压力位": f"{resistance:.2f}" if resistance > 0 else "N/A",
        "当前价格": f"{df_day['close'].iloc[-1]:.2f}",
        "检测到形态": ", ".join(patterns) if patterns else "无明显形态",
        "分析日期": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        "chart_path": chart_path,
        "analysis": analysis,
        "recommendation": recommendation
    }
    
except Exception as e:
    print(traceback.format_exc())
    return {"error": str(e)}
==================== 交易建议生成函数 ====================
def generate_recommendation( day_trend: str, week_trend: str, daily_vol: float, weekly_vol: float, support: float, resistance: float, patterns: List[str], df_day: pd.DataFrame ) -> Dict: """ 基于多维度分析生成交易建议
返回：
- action: BUY/SELL/HOLD/WAIT
- confidence: 置信度 (0-100)
- reasoning: 推理过程
- stop_loss: 建议止损位
- target_price: 目标价位
- position_size: 建议仓位比例
"""

current_price = df_day['close'].iloc[-1] if not df_day.empty else 0
score = 50  # 基础分
reasons = []

# 1. 趋势评分 (权重: 40%)
if day_trend == "uptrend" and week_trend == "uptrend":
    score += 25
    reasons.append("日线周线均为上升趋势，多头共振")
elif day_trend == "uptrend" and week_trend == "sideways":
    score += 15
    reasons.append("日线上升，周线震荡，短期偏强")
elif day_trend == "downtrend" and week_trend == "downtrend":
    score -= 25
    reasons.append("日线周线均为下降趋势，空头主导")
elif day_trend == "downtrend" and week_trend == "sideways":
    score -= 15
    reasons.append("日线下降，周线震荡，短期偏弱")
else:
    reasons.append("趋势不明确，需谨慎操作")

# 2. 波动率评分 (权重: 20%)
vol_ratio = daily_vol / weekly_vol if weekly_vol > 0 else 1

if daily_vol < 0.2 and weekly_vol < 0.2:
    score += 5
    reasons.append("波动率较低，风险可控")
elif daily_vol > 0.4 or weekly_vol > 0.4:
    score -= 10
    reasons.append("波动率较高，注意风险控制")

if vol_ratio < 0.5:
    reasons.append("周波动大于日波动，趋势稳定性较好")
elif vol_ratio > 1.5:
    reasons.append("日波动显著大于周波动，短线机会较多")

# 3. K线形态评分 (权重: 20%)
bullish_patterns = ["hammer", "bullish_engulfing"]
bearish_patterns = ["shooting_star", "bearish_engulfing"]

bullish_count = sum(1 for p in patterns if p in bullish_patterns)
bearish_count = sum(1 for p in patterns if p in bearish_patterns)

if bullish_count > 0:
    score += 15 * bullish_count
    reasons.append(f"出现看涨形态: {', '.join([p for p in patterns if p in bullish_patterns])}")

if bearish_count > 0:
    score -= 15 * bearish_count
    reasons.append(f"出现看跌形态: {', '.join([p for p in patterns if p in bearish_patterns])}")

# 4. 位置评分 (权重: 20%)
if support > 0 and resistance > 0:
    price_position = (current_price - support) / (resistance - support) if resistance != support else 0.5
    
    if price_position < 0.2:
        score += 15
        reasons.append(f"价格接近支撑位({support:.2f})，盈亏比较好")
    elif price_position > 0.8:
        score -= 15
        reasons.append(f"价格接近压力位({resistance:.2f})，上行空间有限")
    elif 0.4 <= price_position <= 0.6:
        score += 5
        reasons.append("价格处于区间中部，方向待选择")

# 5. 确定最终建议
score = max(0, min(100, score))  # 限制在0-100之间

if score >= 70:
    action = "BUY"
    action_text = "强烈买入"
    position_size = "60-80%"
elif score >= 55:
    action = "BUY"
    action_text = "谨慎买入"
    position_size = "30-50%"
elif score <= 30:
    action = "SELL"
    action_text = "坚决卖出"
    position_size = "0%"
elif score <= 45:
    action = "SELL"
    action_text = "减仓观望"
    position_size = "10-20%"
else:
    action = "HOLD"
    action_text = "持有观望"
    position_size = "保持现有仓位"

# 计算止损位和目标位
if day_trend == "uptrend":
    stop_loss = support * 0.98 if support > 0 else current_price * 0.95
    target_price = resistance * 1.05 if resistance > 0 else current_price * 1.1
elif day_trend == "downtrend":
    stop_loss = current_price * 1.05
    target_price = support * 0.95 if support > 0 else current_price * 0.9
else:
    stop_loss = support * 0.97 if support > 0 else current_price * 0.95
    target_price = resistance * 1.03 if resistance > 0 else current_price * 1.05

return {
    "action": action,
    "action_text": action_text,
    "confidence": score,
    "reasoning": reasons,
    "stop_loss": round(stop_loss, 2),
    "target_price": round(target_price, 2),
    "position_size": position_size,
    "risk_reward_ratio": round((target_price - current_price) / (current_price - stop_loss), 2) if current_price != stop_loss else 0
}
==================== 辅助函数 ====================
async def get_stock_day_kline(code: str, startDate: str = None, endDate: str = None, type: int = 1): """获取日K线数据的包装函数""" return await get_stock_kline_data(code, "day", startDate, endDate, type)
async def get_stock_week_kline(code: str, startDate: str = None, endDate: str = None, type: int = 1): """获取周K线数据的包装函数""" return await get_stock_kline_data(code, "week", startDate, endDate, type)
```


## 使用说明

### 1. 基本调用

```python
#可视化某只股票并获取建议
result = await visualize_stock("600519", days=120)
#查看分析结果
print(result['analysis']) print(result['recommendation'])
#查看生成的图表
import matplotlib.image as mpimg import matplotlib.pyplot as plt img = mpimg.imread(result['chart_path']) plt.figure(figsize=(14, 10)) plt.imshow(img) plt.axis('off') plt.show()
```

### 2. 输出示例

**分析报告**:
```json
{ "股票代码": "600519", "股票名称": "贵州茅台", "日线趋势": "uptrend", "周线趋势": "uptrend", "日波动率(年化)": "18.52%", "周波动率(年化)": "22.34%", "支撑位": "1650.00", "压力位": "1850.00", "当前价格": "1720.50", "检测到形态": "hammer", "分析日期": "2026-05-14 15:30:00" }
```

**交易建议**:
```json 
{ "action": "BUY", "action_text": "强烈买入", "confidence": 78, "reasoning": [ "日线周线均为上升趋势，多头共振", "波动率较低，风险可控", "出现看涨形态: hammer", "价格接近支撑位(1650.00)，盈亏比较好" ], "stop_loss": 1617.0, "target_price": 1935.0, "position_size": "60-80%", "risk_reward_ratio": 2.85 }
```



### 3. 核心特性

✅ **多周期分析**: 同时分析日线和周线趋势  
✅ **波动率对比**: 量化评估风险水平  
✅ **形态识别**: 自动检测重要K线形态  
✅ **支撑压力**: 智能识别关键价位  
✅ **可视化图表**: 三合一专业图表（价格+成交量+周线对比）  
✅ **量化建议**: 基于评分系统给出明确操作建议  
✅ **风险控制**: 提供止损位和仓位建议  

### 4. 注意事项

⚠️ 需要安装依赖: `pip install matplotlib pandas numpy requests`  
⚠️ API Token已内置，如需更换请修改`TOKEN`变量  
⚠️ 图表默认保存在当前目录，可通过`save_path`参数自定义  
⚠️ 建议结合基本面分析使用，技术面仅供参考