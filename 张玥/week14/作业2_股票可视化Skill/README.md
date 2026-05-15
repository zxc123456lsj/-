# 作业 2：股票波动可视化 Skill

本作业完成一个股票可视化 Skill，对股票日波动和周波动绘制到同一张图中，并根据波动大小给出教学型买入/卖出建议。

## 参考来源

- 股票函数和画图思路参考：`Week12/03_本地股票助手.py`
- Skill 写法参考：`Week14/skills/autostock/SKILL.md`、`Week14/skills/get-news/SKILL.md`

正式 Skill 定义文件位于：

```text
D:\AI_study_env\files\study\Week14\skills\stock-volatility-visualizer\SKILL.md
```

## 运行方式

```powershell
cd D:\AI_study_env\files\study\Week14\homework\作业2_股票可视化Skill
D:\AI_study_env\miniconda3\envs\py312\python.exe stock_visualization_skill.py --csv sample_data\stock_demo.csv
```

指定股票名称：

```powershell
D:\AI_study_env\miniconda3\envs\py312\python.exe stock_visualization_skill.py --csv sample_data\stock_demo.csv --symbol DEMO
```

## 输入数据格式

CSV 至少包含：

```csv
date,close
2026-01-02,100.50
2026-01-05,101.20
```

## 输出内容

程序会输出：

- 数据时间范围
- 最近日波动
- 最近周波动
- 波动最高和最低的日期
- 买入/卖出建议
- 图表文件路径

图表会保存到 `output/` 目录。

## 说明

为了保证作业稳定运行，本实现默认使用本地 CSV 示例数据，不依赖 `yfinance` 或外部网络接口。

> 注意：本作业中的买卖建议仅用于课程演示，不构成真实投资建议。
