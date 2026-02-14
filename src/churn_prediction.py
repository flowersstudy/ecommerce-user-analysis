"""
流失预测模块
使用机器学习预测用户流失概率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
import joblib
import os

def run_churn_model():
    """运行流失预测模型的主要函数"""
    print("🤖 开始流失预测分析...")
    
    # 计算项目根目录路径
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "rfm_results.csv")
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到 RFM 数据文件 {DATA_PATH}")
        print("请先运行 rfm_analysis.py")
        return
    
    # 加载 RFM 数据
    rfm_df = pd.read_csv(DATA_PATH, index_col=0)
    
    print(f"📊 准备流失预测数据...")
    print(f"流失定义: Recency > 60 天")
    
    # 创建流失标签
    rfm_df['Churn'] = (rfm_df['Recency'] > 60).astype(int)
    
    # 特征选择
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[features]
    y = rfm_df['Churn']
    
    print(f"总样本数: {len(X)}")
    print(f"流失用户: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"活跃用户: {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 训练逻辑回归模型
    print("\n🤖 训练逻辑回归模型...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, predictions)
    print(f"模型准确率: {accuracy:.4f}")
    
    print("\n📈 模型评估报告:")
    print("="*50)
    print(classification_report(y_test, predictions, 
                              target_names=['活跃', '流失']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    print("\n混淆矩阵:")
    print(cm)
    
    # 计算各项指标
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"特异度 (Specificity): {specificity:.4f}")
    
    # 训练随机森林模型（对比用）
    print("\n🌲 训练随机森林模型...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    print(f"随机森林准确率: {rf_accuracy:.4f}")
    
    # 特征重要性
    importances = pd.DataFrame({
        'feature': ['Recency', 'Frequency', 'Monetary'],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性:")
    print(importances)
    
    # 为所有用户预测流失风险
    all_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    rfm_df['Churn_Probability'] = model.predict_proba(all_features)[:, 1]
    rfm_df['Churn_Risk_Level'] = pd.cut(
        rfm_df['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['低风险', '中风险', '高风险']
    )
    
    # 保存预测结果
    OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "churn_predictions.csv")
    rfm_df.to_csv(output_path)
    print(f"\n✅ 流失预测结果已保存: {output_path}")
    
    # 统计风险分布
    risk_dist = rfm_df['Churn_Risk_Level'].value_counts()
    print("\n流失风险分布:")
    for level, count in risk_dist.items():
        pct = count / len(rfm_df) * 100
        print(f"  {level}: {count} 人 ({pct:.1f}%)")
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, "churn_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    print("Churn prediction done")


def main():
    """主函数"""
    run_churn_model()


if __name__ == '__main__':
    main()
