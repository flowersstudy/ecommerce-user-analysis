# 🛍️ 电商用户行为分析与流失预测

基于 RFM 模型的电商用户分群、价值分析和流失预测项目。

## 📊 项目简介

本项目使用电商平台交易数据（Online Retail Dataset），通过 **RFM 模型** 对用户进行价值分群，结合 **KMeans 聚类** 和 **逻辑回归** 进行流失预测，为精细化运营提供数据支持。

**核心目标：**
1. 识别高价值用户和流失风险用户
2. 建立流失预测模型，提前预警
3. 提供可落地的运营策略建议

## 🛠️ 技术栈

- **SQL**: 数据查询与清洗（RFM 指标计算）
- **Python**: 
  - pandas / numpy: 数据处理
  - matplotlib / seaborn: 数据可视化
  - scikit-learn: 机器学习（KMeans、逻辑回归）

## 📁 项目结构

```
ecommerce-user-analysis/
├── 📋 README.md                   # 项目说明
├── 📦 requirements.txt            # Python依赖
│
├── 🎨 assets/images/              # 可视化预览图
│   ├── user_distribution.png
│   ├── user_clustering_kmeans.png
│   ├── monetary_distribution.png
│   ├── frequency_distribution.png
│   └── churn_prediction_accuracy.png
│
├── 🐍 src/                        # Python源代码
│   ├── rfm_analysis.py           # RFM模型实现
│   ├── visualization.py          # 可视化模块
│   └── churn_prediction.py       # 流失预测模型
│
├── 🗄️ sql/
│   └── rfm_analysis.sql          # SQL分析脚本
│
└── 📄 reports/
    └── analysis_report.md        # 详细分析报告
```

## 📈 核心分析内容

### 1. RFM 用户分群

| 用户类型 | 数量 | 占比 | 特征 | 运营策略 |
|---------|------|------|------|---------|
| **高价值用户** | ~716 | 16.4% | 最近购买、高频、高消费 | VIP服务、专属优惠 |
| **流失风险用户** | ~738 | 16.9% | 很久未买、低频、低消费 | 召回活动、大额优惠券 |
| **忠实用户** | ~68 | 1.6% | 高频高消费，需唤醒 | 新品优先体验 |
| **新用户** | ~342 | 7.8% | 最近购买但频次低 | 引导复购、会员权益 |

### 2. 流失预测模型

- **算法**: 逻辑回归
- **准确率**: 85.3%
- **关键发现**: Recency 是最重要的预测因子
- **应用**: 对 Recency > 45 天的用户提前预警

### 3. KMeans 聚类

- **聚类数**: 3
- **结果**: 识别出高价值用户簇（8%）、潜力用户簇（20%）、普通用户簇（72%）

---

## 📊 可视化结果

### 用户分群分布

![用户分群分布](assets/images/user_distribution.png)

*高价值用户与流失风险用户数量相当，忠实用户群体较小*

### KMeans 聚类分析

![用户聚类](assets/images/user_clustering_kmeans.png)

*用户在 Frequency 和 Monetary 维度呈现明显的群体分化*

### 消费金额分布

![消费金额分布](assets/images/monetary_distribution.png)

*不同用户群体的消费金额差异显著，高价值用户有明显的长尾特征*

### 购买频次分布

![购买频次分布](assets/images/frequency_distribution.png)

*大多数用户购买频次集中在 1-10 次，高价值用户频次更高*

### 流失预测模型

![模型准确率](assets/images/churn_prediction_accuracy.png)

*逻辑回归模型准确率 85%+，可用于流失预警*

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包: pandas, numpy, matplotlib, seaborn, scikit-learn

### 安装

```bash
pip install -r requirements.txt
```

### 运行分析

```bash
# 1. RFM 分析
python src/rfm_analysis.py

# 2. 流失预测
python src/churn_prediction.py
```

---

## 💡 业务建议

### 短期行动（1-2周）
- **流失预警**: 对 Recency > 45 天的用户发送召回邮件
- **高价值维护**: 为 RFM=555 用户提供专属客服通道
- **新用户转化**: 首购后 7 天内推送复购优惠

### 中期策略（1-3个月）
1. 建立用户标签体系，自动化 RFM 计算
2. A/B 测试不同召回策略效果
3. 设计会员阶梯成长路径

### 长期规划（3-6个月）
1. 引入 XGBoost 优化预测模型
2. 搭建个性化推荐系统
3. 建立完整的用户生命周期管理体系

---

## 📝 SQL 能力

项目包含完整的 SQL 分析脚本 (`sql/rfm_analysis.sql`)，展示：
- RFM 指标计算（使用 NTILE 窗口函数）
- 用户分群统计
- 流失风险识别
- 月度销售趋势分析
- 复购率计算

---

## 📄 数据说明

- **数据集**: Online Retail Dataset (UCI Machine Learning Repository)
- **时间跨度**: 2010-12-01 至 2011-12-09
- **记录数**: 541,909 条交易记录
- **用户数**: 4,372 名独立用户
- **数据字段**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## 👤 作者

**世茵**  
湘潭大学 · 信息管理与信息系统  
📧 1501232462@qq.com

---

## 📜 License

MIT License
