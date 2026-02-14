import pandas as pd

# 加载 CSV 文件
data = pd.read_csv('data.csv', encoding='latin1')

# 查看数据的前几行
print(data.head())

print(data.shape)
print(data.columns)
print(data.info())

import pandas as pd

data = pd.read_csv('data.csv', encoding='latin1')

# 1. 删除没有用户ID的行（游客无法做用户分析）
data = data.dropna(subset=['CustomerID'])

# 2. 转换时间格式
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# 3. 计算金额 = 数量 × 单价
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']


print(data.head())
print(data.shape)

# 获取数据最后一天（作为当前时间）
snapshot_date = data['InvoiceDate'].max()
print("数据最后日期:", snapshot_date)

# 按用户分组计算RFM
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # R
    'InvoiceNo': 'nunique',                                   # F
    'TotalPrice': 'sum'                                       # M
})

# 改列名
rfm.columns = ['Recency', 'Frequency', 'Monetary']

print(rfm.head())
print("用户数：", rfm.shape)

# 使用 3 个分位数，并且加入 duplicates='drop' 来处理重复的切割点
rfm['R_Score'] = pd.qcut(rfm['Recency'], 3, labels=[3, 2, 1], duplicates='drop')  # Recency 越小越好
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 3, labels=[1, 2, 3], duplicates='drop')  # Frequency 越大越好
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 3, labels=[1, 2, 3], duplicates='drop')  # Monetary 越大越好

# 创建一个 RFM 得分列
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# 查看前几行数据
print(rfm.head())

# 高价值用户：RFM 分数为 333
high_value_users = rfm[rfm['RFM_Score'] == '333']
print("高价值用户数:", high_value_users.shape[0])

# 潜在流失用户：RFM 分数为 111
churn_users = rfm[rfm['RFM_Score'] == '111']
print("潜在流失用户数:", churn_users.shape[0])

# 忠实用户（频繁购买且消费高，但不一定最近购买）
loyal_users = rfm[rfm['RFM_Score'] == '133']
print("忠实用户数:", loyal_users.shape[0])

# 新用户（最近购买且少量购买）
new_users = rfm[rfm['RFM_Score'].isin(['311', '312', '313'])]
print("新用户数:", new_users.shape[0])

import matplotlib.pyplot as plt
import os

# 确认当前工作目录
print("当前工作目录:", os.getcwd())  # 检查当前工作目录

# 1. 用户分布柱状图
user_counts = {
    'High Value Users': high_value_users.shape[0],
    'Churn Users': churn_users.shape[0],
    'Loyal Users': loyal_users.shape[0],
    'New Users': new_users.shape[0]
}

# 创建保存文件夹（如果不存在）
save_path = 'D:/study-/大三寒假/data_analysis/'
os.makedirs(save_path, exist_ok=True)

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(user_counts.keys(), user_counts.values(), color=['green', 'orange', 'blue', 'red'])
plt.title('User Distribution', fontsize=14)
plt.xlabel('User Type', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)

# 保存图形到指定路径
plt.savefig(os.path.join(save_path, 'user_distribution.png'), dpi=300)  # 保存图像到指定目录
plt.show()  # 显示图表

# 2. 绘制不同用户群体的消费金额分布（箱型图）
plt.figure(figsize=(10, 6))
plt.boxplot([high_value_users['Monetary'], churn_users['Monetary'], loyal_users['Monetary'], new_users['Monetary']],
            labels=['High Value Users', 'Churn Users', 'Loyal Users', 'New Users'], notch=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            flierprops=dict(marker='o', color='red', markersize=5))
plt.title('Monetary Value Distribution Across User Groups', fontsize=14)
plt.xlabel('User Group', fontsize=12)
plt.ylabel('Monetary Value', fontsize=12)

# 保存图形到指定路径
plt.savefig(os.path.join(save_path, 'monetary_distribution.png'), dpi=300)  # 保存图像到指定目录
plt.show()  # 显示图表

# 3. 绘制不同用户群体的购买频次分布（直方图）
plt.figure(figsize=(10, 6))
plt.hist([high_value_users['Frequency'], churn_users['Frequency'], loyal_users['Frequency'], new_users['Frequency']],
         bins=20, label=['High Value Users', 'Churn Users', 'Loyal Users', 'New Users'], alpha=0.7)
plt.title('Purchase Frequency Distribution Across User Groups', fontsize=14)
plt.xlabel('Purchase Frequency', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.legend()

# 保存图形到指定路径
plt.savefig(os.path.join(save_path, 'frequency_distribution.png'), dpi=300)  # 保存图像到指定目录
plt.show()  # 显示图表

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# # # 用户流失预测 和 聚类分析（使用 KMeans）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import os

# 创建保存文件夹（如果不存在）
save_path = 'D:/study-/大三寒假/data_analysis/'
os.makedirs(save_path, exist_ok=True)

# 1. 创建流失标签（假设 Recency > 60 天的用户为流失用户）
rfm['Churn'] = np.where(rfm['Recency'] > 60, 1, 0)  # 1 表示流失，0 表示活跃用户

# 2. 用户流失预测：训练逻辑回归模型
X = rfm[['Recency', 'Frequency', 'Monetary']]  # 特征
y = rfm['Churn']  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率: ", accuracy)
print(classification_report(y_test, y_pred))  # 显示更详细的分类报告

# 3. KMeans 聚类分析：对用户进行聚类
X_cluster = rfm[['Recency', 'Frequency', 'Monetary']]  # 使用这些特征来聚类

# 使用 KMeans 聚类，设置聚类数为 3
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_cluster)

# 查看聚类结果
print(rfm[['Recency', 'Frequency', 'Monetary', 'Cluster']].head())

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis')

# 设置图表的标题、X 轴和 Y 轴的标签
plt.title('KMeans Clustering: Users Based on Recency, Frequency, and Monetary', fontsize=14)
plt.xlabel('Frequency', fontsize=12)  # X轴标签：购买频次
plt.ylabel('Monetary', fontsize=12)  # Y轴标签：消费金额

# 保存图形到指定路径
plt.savefig(os.path.join(save_path, 'user_clustering_kmeans.png'), dpi=300)  # 使用完整路径保存图像
plt.show()  # 显示图表

# 4. 流失预测模型结果的图表（例如：混淆矩阵，分类报告，或者准确率的条形图）

# 显示模型准确率的条形图
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.title('Churn Prediction Model Accuracy', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# 保存准确率图表
plt.savefig(os.path.join(save_path, 'churn_prediction_accuracy.png'), dpi=300)  # 使用完整路径保存图像
plt.show()  # 显示图表
