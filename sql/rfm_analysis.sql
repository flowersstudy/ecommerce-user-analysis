-- ============================================================
-- 电商用户 RFM 分析 SQL 脚本
-- 数据库: 支持 MySQL / PostgreSQL / SQLite
-- 说明: 展示如何用 SQL 实现 RFM 分析
-- ============================================================

-- ============================================================
-- 1. 数据准备
-- ============================================================

-- 创建表结构
CREATE TABLE IF NOT EXISTS transactions (
    InvoiceNo VARCHAR(20),
    StockCode VARCHAR(20),
    Description VARCHAR(255),
    Quantity INT,
    InvoiceDate TIMESTAMP,
    UnitPrice DECIMAL(10,2),
    CustomerID INT,
    Country VARCHAR(50)
);

-- ============================================================
-- 2. 数据清洗视图
-- ============================================================

-- 创建清洗后的数据视图
CREATE VIEW clean_transactions AS
SELECT 
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    InvoiceDate,
    UnitPrice,
    CustomerID,
    Country,
    (Quantity * UnitPrice) AS TotalPrice
FROM transactions
WHERE CustomerID IS NOT NULL
  AND Quantity > 0
  AND UnitPrice > 0;

-- ============================================================
-- 3. RFM 指标计算
-- ============================================================

-- 计算每个用户的 RFM 指标
WITH user_rfm AS (
    SELECT 
        CustomerID,
        -- Recency: 距离最后购买的天数
        DATEDIFF(CURRENT_DATE, MAX(InvoiceDate)) AS Recency,
        -- Frequency: 购买订单数（去重）
        COUNT(DISTINCT InvoiceNo) AS Frequency,
        -- Monetary: 总消费金额
        SUM(Quantity * UnitPrice) AS Monetary
    FROM clean_transactions
    GROUP BY CustomerID
),

-- ============================================================
-- 4. RFM 评分计算
-- ============================================================

rfm_scores AS (
    SELECT 
        CustomerID,
        Recency,
        Frequency,
        Monetary,
        -- R 评分: Recency 越小分数越高 (5分制)
        NTILE(5) OVER (ORDER BY Recency DESC) AS R_Score,
        -- F 评分: Frequency 越大分数越高
        NTILE(5) OVER (ORDER BY Frequency ASC) AS F_Score,
        -- M 评分: Monetary 越大分数越高
        NTILE(5) OVER (ORDER BY Monetary ASC) AS M_Score
    FROM user_rfm
),

-- ============================================================
-- 5. 用户分群
-- ============================================================

user_segments AS (
    SELECT 
        CustomerID,
        Recency,
        Frequency,
        Monetary,
        R_Score,
        F_Score,
        M_Score,
        CONCAT(R_Score, F_Score, M_Score) AS RFM_Score,
        CASE 
            WHEN R_Score >= 4 AND F_Score >= 4 AND M_Score >= 4 THEN '高价值用户'
            WHEN R_Score <= 2 AND F_Score <= 2 AND M_Score <= 2 THEN '流失风险用户'
            WHEN F_Score >= 4 AND M_Score >= 4 THEN '忠实用户'
            WHEN R_Score >= 4 AND F_Score <= 2 THEN '新用户'
            ELSE '其他用户'
        END AS Segment
    FROM rfm_scores
)

-- ============================================================
-- 6. 查询结果
-- ============================================================

-- 6.1 查看 RFM 结果
SELECT * FROM user_segments
ORDER BY Monetary DESC
LIMIT 100;

-- ============================================================
-- 7. 分群统计分析
-- ============================================================

-- 7.1 各群体用户数量统计
SELECT 
    Segment,
    COUNT(*) AS UserCount,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS Percentage
FROM user_segments
GROUP BY Segment
ORDER BY UserCount DESC;

-- 7.2 各群体 RFM 均值统计
SELECT 
    Segment,
    COUNT(*) AS UserCount,
    ROUND(AVG(Recency), 2) AS AvgRecency,
    ROUND(AVG(Frequency), 2) AS AvgFrequency,
    ROUND(AVG(Monetary), 2) AS AvgMonetary,
    ROUND(SUM(Monetary), 2) AS TotalMonetary
FROM user_segments
GROUP BY Segment
ORDER BY TotalMonetary DESC;

-- ============================================================
-- 8. 流失分析
-- ============================================================

-- 8.1 流失用户识别 (Recency > 60 天)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    Segment,
    CASE 
        WHEN Recency > 180 THEN '严重流失'
        WHEN Recency > 90 THEN '高度风险'
        WHEN Recency > 60 THEN '中度风险'
        ELSE '正常'
    END AS ChurnRisk
FROM user_segments
WHERE Recency > 60
ORDER BY Recency DESC;

-- 8.2 流失用户统计
SELECT 
    CASE 
        WHEN Recency > 180 THEN '严重流失'
        WHEN Recency > 90 THEN '高度风险'
        WHEN Recency > 60 THEN '中度风险'
        ELSE '正常'
    END AS ChurnRisk,
    COUNT(*) AS UserCount,
    ROUND(AVG(Monetary), 2) AS AvgMonetary,
    ROUND(SUM(Monetary), 2) AS TotalMonetary
FROM user_segments
GROUP BY 
    CASE 
        WHEN Recency > 180 THEN '严重流失'
        WHEN Recency > 90 THEN '高度风险'
        WHEN Recency > 60 THEN '中度风险'
        ELSE '正常'
    END
ORDER BY 
    CASE 
        WHEN ChurnRisk = '严重流失' THEN 1
        WHEN ChurnRisk = '高度风险' THEN 2
        WHEN ChurnRisk = '中度风险' THEN 3
        ELSE 4
    END;

-- ============================================================
-- 9. 销售趋势分析
-- ============================================================

-- 9.1 月度销售统计
SELECT 
    DATE_FORMAT(InvoiceDate, '%Y-%m') AS Month,
    COUNT(DISTINCT InvoiceNo) AS OrderCount,
    COUNT(DISTINCT CustomerID) AS CustomerCount,
    SUM(Quantity * UnitPrice) AS Revenue,
    AVG(Quantity * UnitPrice) AS AvgOrderValue
FROM clean_transactions
GROUP BY DATE_FORMAT(InvoiceDate, '%Y-%m')
ORDER BY Month;

-- 9.2 每日活跃用户 (DAU) 趋势
SELECT 
    DATE(InvoiceDate) AS Date,
    COUNT(DISTINCT CustomerID) AS DAU,
    COUNT(DISTINCT InvoiceNo) AS Orders,
    SUM(Quantity * UnitPrice) AS DailyRevenue
FROM clean_transactions
GROUP BY DATE(InvoiceDate)
ORDER BY Date;

-- ============================================================
-- 10. 商品分析
-- ============================================================

-- 10.1 热销商品 TOP 20
SELECT 
    StockCode,
    Description,
    SUM(Quantity) AS TotalQuantity,
    SUM(Quantity * UnitPrice) AS TotalRevenue,
    COUNT(DISTINCT CustomerID) AS UniqueCustomers
FROM clean_transactions
GROUP BY StockCode, Description
ORDER BY TotalQuantity DESC
LIMIT 20;

-- 10.2 商品品类销售统计 (按描述关键词)
SELECT 
    CASE 
        WHEN Description LIKE '%HEART%' THEN 'Heart系列'
        WHEN Description LIKE '%CANDLE%' THEN '蜡烛系列'
        WHEN Description LIKE '%LIGHT%' THEN '灯具系列'
        WHEN Description LIKE '%BAG%' THEN '包袋系列'
        WHEN Description LIKE '%BOX%' THEN '盒子系列'
        ELSE '其他'
    END AS Category,
    COUNT(*) AS SalesCount,
    SUM(Quantity) AS TotalQuantity,
    SUM(Quantity * UnitPrice) AS TotalRevenue
FROM clean_transactions
GROUP BY 
    CASE 
        WHEN Description LIKE '%HEART%' THEN 'Heart系列'
        WHEN Description LIKE '%CANDLE%' THEN '蜡烛系列'
        WHEN Description LIKE '%LIGHT%' THEN '灯具系列'
        WHEN Description LIKE '%BAG%' THEN '包袋系列'
        WHEN Description LIKE '%BOX%' THEN '盒子系列'
        ELSE '其他'
    END
ORDER BY TotalRevenue DESC;

-- ============================================================
-- 11. 用户购买行为分析
-- ============================================================

-- 11.1 用户购买频次分布
SELECT 
    CASE 
        WHEN Frequency = 1 THEN '1次'
        WHEN Frequency BETWEEN 2 AND 5 THEN '2-5次'
        WHEN Frequency BETWEEN 6 AND 10 THEN '6-10次'
        WHEN Frequency BETWEEN 11 AND 20 THEN '11-20次'
        ELSE '20次以上'
    END AS FrequencyGroup,
    COUNT(*) AS UserCount,
    ROUND(AVG(Monetary), 2) AS AvgMonetary
FROM user_rfm
GROUP BY 
    CASE 
        WHEN Frequency = 1 THEN '1次'
        WHEN Frequency BETWEEN 2 AND 5 THEN '2-5次'
        WHEN Frequency BETWEEN 6 AND 10 THEN '6-10次'
        WHEN Frequency BETWEEN 11 AND 20 THEN '11-20次'
        ELSE '20次以上'
    END
ORDER BY 
    CASE 
        WHEN FrequencyGroup = '1次' THEN 1
        WHEN FrequencyGroup = '2-5次' THEN 2
        WHEN FrequencyGroup = '6-10次' THEN 3
        WHEN FrequencyGroup = '11-20次' THEN 4
        ELSE 5
    END;

-- 11.2 复购率计算
WITH user_purchase_count AS (
    SELECT 
        CustomerID,
        COUNT(DISTINCT InvoiceNo) AS PurchaseCount
    FROM clean_transactions
    GROUP BY CustomerID
)
SELECT 
    COUNT(CASE WHEN PurchaseCount > 1 THEN 1 END) AS RepeatCustomers,
    COUNT(*) AS TotalCustomers,
    ROUND(
        COUNT(CASE WHEN PurchaseCount > 1 THEN 1 END) * 100.0 / COUNT(*), 
        2
    ) AS RepeatRate
FROM user_purchase_count;

-- ============================================================
-- 12. 留存分析
-- ============================================================

-- 12.1 计算首次购买和最后购买时间
SELECT 
    CustomerID,
    MIN(InvoiceDate) AS FirstPurchase,
    MAX(InvoiceDate) AS LastPurchase,
    DATEDIFF(MAX(InvoiceDate), MIN(InvoiceDate)) AS CustomerLifespan,
    COUNT(DISTINCT InvoiceNo) AS TotalOrders,
    SUM(Quantity * UnitPrice) AS TotalSpent
FROM clean_transactions
GROUP BY CustomerID
HAVING COUNT(DISTINCT InvoiceNo) > 1
ORDER BY TotalSpent DESC
LIMIT 50;

-- ============================================================
-- 13. 国家/地区分析
-- ============================================================

-- 13.1 各国销售统计
SELECT 
    Country,
    COUNT(DISTINCT CustomerID) AS CustomerCount,
    COUNT(DISTINCT InvoiceNo) AS OrderCount,
    SUM(Quantity * UnitPrice) AS TotalRevenue,
    ROUND(AVG(Quantity * UnitPrice), 2) AS AvgOrderValue
FROM clean_transactions
GROUP BY Country
HAVING CustomerCount >= 10
ORDER BY TotalRevenue DESC
LIMIT 20;

-- ============================================================
-- 使用说明:
-- 1. 根据实际数据库类型调整日期函数 (DATEDIFF, DATE_FORMAT 等)
-- 2. MySQL 和 PostgreSQL 语法略有差异
-- 3. 对于大数据量，建议在 CustomerID 和 InvoiceDate 上创建索引
-- ============================================================
