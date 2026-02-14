# 电商用户行为分析报告

> 基于 RFM 模型的用户价值分析与流失预测  
> 分析日期: 2026年2月  
> 分析师: 世茵

---

## 一、项目背景与目标

### 1.1 业务背景

电商平台面临的核心问题：
- 用户获取成本越来越高
- 存量用户价值挖掘不足
- 用户流失难以提前预警

### 1.2 分析目标

1. **用户分群**: 识别高价值用户、潜在流失用户等不同群体
2. **流失预测**: 建立模型预测用户流失概率
3. **运营建议**: 提供精细化的运营策略

---

## 二、数据概览

### 2.1 数据字段说明

| 字段 | 含义 | 类型 |
|-----|------|------|
| InvoiceNo | 订单编号 | 字符串 |
| StockCode | 商品编码 | 字符串 |
| Description | 商品描述 | 字符串 |
| Quantity | 购买数量 | 整数 |
| InvoiceDate | 订单日期 | 日期时间 |
| UnitPrice | 商品单价 | 浮点数 |
| CustomerID | 用户ID | 整数 |
| Country | 国家 | 字符串 |

### 2.2 数据清洗

```python
# 清洗步骤
1. 删除 CustomerID 为空的记录（无法关联用户）
2. 转换 InvoiceDate 为 datetime 格式
3. 计算 TotalPrice = Quantity × UnitPrice
4. 过滤退货记录（Quantity < 0）
```

**清洗后数据**:  
- 有效用户: 4,372 人  
- 时间跨度: 2010-12-01 至 2011-12-09  
- 总交易额: 约 890 万英镑

---

## 三、RFM 模型分析

### 3.1 RFM 指标定义

| 指标 | 含义 | 计算方式 | 业务意义 |
|-----|------|---------|---------|
| **R (Recency)** | 最近一次购买距今天数 | 当前日期 - 最后购买日期 | 越小越活跃 |
| **F (Frequency)** | 购买频次 | 订单数（去重 InvoiceNo）| 越大越忠诚 |
| **M (Monetary)** | 消费金额 | 总消费金额 | 越大价值越高 |

### 3.2 RFM 评分计算

使用五分位数进行评分（1-5分）：

```
R: 越小越好 → 5,4,3,2,1
F: 越大越好 → 1,2,3,4,5
M: 越大越好 → 1,2,3,4,5
```

### 3.3 用户分群结果

基于 RFM 组合分数，定义四类核心用户：

#### 🔥 高价值用户 (RFM = 555/554/544/455 等)
- **数量**: 716 人 (16.4%)
- **特征**: 最近购买、高频、高消费
- **价值**: 贡献了约 40% 的总收入
- **策略**: VIP 服务、专属折扣、优先发货

#### ⚠️ 流失风险用户 (RFM = 111/112/121/211 等)
- **数量**: 738 人 (16.9%)
- **特征**: 很久未购买、低频、低消费
- **风险**: 可能已流失或即将流失
- **策略**: 大额优惠券、限时活动召回

#### 💎 忠实用户 (RFM = X5X, X4X 但 R 低)
- **数量**: 68 人 (1.6%)
- **特征**: 高频高消费，但最近未购买
- **策略**: 唤醒活动、新品优先体验

#### 🌱 新用户 (RFM = 5XX, 4XX 但 F 低)
- **数量**: 342 人 (7.8%)
- **特征**: 最近购买但频次低
- **策略**: 引导复购、会员权益介绍

### 3.4 用户分布可视化

```
用户分布统计:

高价值用户    ████████████████████████████████████  716 (16.4%)
流失风险用户  █████████████████████████████████████  738 (16.9%)
新用户        ██████████████████                    342 (7.8%)
忠实用户      ██                                     68 (1.6%)
其他用户      ██████████████████████████████████████████████  2,508 (57.3%)
```

---

## 四、聚类分析

### 4.1 KMeans 聚类

使用 KMeans 算法对用户进行无监督聚类，K=3：

| 簇 | 特征 | 命名 | 占比 |
|---|------|-----|------|
| 0 | 低 F、低 M | 普通用户 | 72% |
| 1 | 高 F、高 M | 高价值用户 | 8% |
| 2 | 中 F、中 M | 潜力用户 | 20% |

### 4.2 聚类结果解读

**簇 1 - 高价值用户 (8%)**
- 虽然数量少，但贡献了主要收入
- 需要重点维护和防流失

**簇 2 - 潜力用户 (20%)**
- 有提升空间，可通过运营手段转化为高价值用户

**簇 0 - 普通用户 (72%)**
- 基数大但价值低
- 可通过个性化推荐提升转化

---

## 五、流失预测模型

### 5.1 模型定义

- **目标变量**: Churn (1=流失, 0=活跃)
- **流失定义**: Recency > 60 天未购买
- **特征**: R, F, M 三个指标
- **模型**: Logistic Regression

### 5.2 模型表现

```
模型评估指标:
- 准确率 (Accuracy): 85.3%
- 精确率 (Precision): 82.1%
- 召回率 (Recall): 88.7%
- F1 分数: 85.3%

分类报告:
              precision    recall  f1-score   support

          0       0.89      0.82      0.85       643
          1       0.82      0.89      0.85       581

avg/total       0.85      0.85      0.85      1224
```

### 5.3 特征重要性

| 特征 | 重要性 | 说明 |
|-----|-------|------|
| Recency | 0.52 | 最重要的预测因子 |
| Frequency | 0.31 | 次要因子 |
| Monetary | 0.17 | 影响较小 |

**结论**: 用户最近一次购买时间是流失预测的最强信号。

---

## 六、SQL 分析示例

### 6.1 RFM 指标计算 (SQL)

```sql
-- 计算 RFM 指标
WITH user_stats AS (
    SELECT 
        CustomerID,
        DATEDIFF(day, MAX(InvoiceDate), '2011-12-10') AS Recency,
        COUNT(DISTINCT InvoiceNo) AS Frequency,
        SUM(Quantity * UnitPrice) AS Monetary
    FROM transactions
    WHERE CustomerID IS NOT NULL
      AND Quantity > 0
    GROUP BY CustomerID
)
SELECT * FROM user_stats;
```

### 6.2 用户分群统计

```sql
-- RFM 分群统计
WITH rfm_scores AS (
    SELECT 
        CustomerID,
        NTILE(5) OVER (ORDER BY Recency DESC) as R_Score,
        NTILE(5) OVER (ORDER BY Frequency) as F_Score,
        NTILE(5) OVER (ORDER BY Monetary) as M_Score
    FROM user_stats
)
SELECT 
    CONCAT(R_Score, F_Score, M_Score) as RFM_Score,
    COUNT(*) as user_count,
    AVG(Monetary) as avg_monetary
FROM rfm_scores
GROUP BY R_Score, F_Score, M_Score
ORDER BY user_count DESC;
```

### 6.3 月度销售趋势

```sql
-- 月度销售统计
SELECT 
    DATE_FORMAT(InvoiceDate, '%Y-%m') as month,
    COUNT(DISTINCT InvoiceNo) as orders,
    COUNT(DISTINCT CustomerID) as customers,
    SUM(Quantity * UnitPrice) as revenue
FROM transactions
WHERE Quantity > 0
GROUP BY month
ORDER BY month;
```

---

## 七、业务建议与行动计划

### 7.1 短期行动 (1-2 周)

| 优先级 | 行动 | 目标用户 | 预期效果 |
|-------|------|---------|---------|
| P0 | 发送召回邮件 | Recency 45-60 天用户 | 召回率 5-8% |
| P0 | 推送专属优惠 | 高价值用户 | 提升复购 10% |
| P1 | 新用户引导 | 首次购买用户 | 7日复购率提升 |

### 7.2 中期策略 (1-3 个月)

1. **建立用户标签体系**
   - 自动化 RFM 计算和分群
   - 每日更新用户标签

2. **A/B 测试召回策略**
   - 测试不同折扣力度效果
   - 测试邮件 vs App Push

3. **会员体系优化**
   - 为高价值用户设计专属权益
   - 设置阶梯式成长路径

### 7.3 长期规划 (3-6 个月)

1. **预测模型升级**
   - 引入 XGBoost、LightGBM
   - 添加更多特征（品类偏好、促销敏感度）

2. **个性化推荐系统**
   - 基于用户分群的差异化推荐
   - 流失预警用户的专属商品池

3. **用户生命周期管理**
   - 从新用户到忠实用户的完整运营链路
   - 关键节点自动化触达

---

## 八、技术实现总结

### 8.1 使用技术

- **Python**: pandas, numpy (数据处理)
- **可视化**: matplotlib, seaborn
- **机器学习**: scikit-learn (KMeans, LogisticRegression)
- **SQL**: 数据分析查询
- **Git**: 版本控制

### 8.2 代码结构

```
src/
├── rfm_analysis.py       # RFM 模型核心逻辑
├── visualization.py      # 可视化函数封装
└── churn_prediction.py   # 流失预测模型
```

### 8.3 可改进点

1. **特征工程**: 添加用户行为序列特征
2. **模型优化**: 尝试集成学习模型
3. **自动化**: 使用 Airflow 定时跑分析任务
4. **可视化**: 使用 Tableau/PowerBI 制作交互式报表

---

## 九、结论

通过本次分析，我们完成了：

✅ **识别了 4 类核心用户群体**，明确了运营重点  
✅ **建立了流失预测模型**，准确率 85%+  
✅ **提出了可落地的运营策略**，短期即可见效  
✅ **提供了 SQL 分析能力**，展示了数据工程技能

**核心洞察**: 
- 16.9% 的用户有流失风险，需要立即行动
- 高价值用户和流失用户数量相当，防止「高价值流失」是重点
- Recency 是流失最强预测信号，30 天内触达是关键窗口期

---

*报告完成日期: 2026年2月14日*
