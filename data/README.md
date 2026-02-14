# 数据目录

此目录用于存放项目所需的原始数据文件。

## 所需文件

**文件名**: `data.csv`

**来源**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)

**替代下载**:
- Kaggle: https://www.kaggle.com/datasets/carrie1/ecommerce-data
- 直接链接: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

## 数据说明

- **记录数**: 541,909 条交易记录
- **时间跨度**: 2010-12-01 至 2011-12-09
- **字段**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## 注意事项

⚠️ **请勿将此数据文件提交到 GitHub！**

数据文件已通过 `.gitignore` 排除在版本控制之外，因为：
1. 文件较大（约 40MB），不适合 Git 管理
2. 这是原始数据，不需要版本追踪
3. 可以从公开渠道重新下载

## 使用方式

将下载的 `data.csv` 文件放置于此目录，代码会自动读取。
