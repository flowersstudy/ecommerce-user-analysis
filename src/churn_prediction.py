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

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ChurnPredictor:
    """用户流失预测器"""
    
    def __init__(self, rfm_df):
        """
        初始化预测器
        
        Args:
            rfm_df: 包含 RFM 数据的 DataFrame
        """
        self.rfm = rfm_df.copy()
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.predictions = None
        self.probabilities = None
        
    def prepare_data(self, churn_threshold=60, test_size=0.3):
        """
        准备训练和测试数据
        
        Args:
            churn_threshold: 定义流失的 Recency 阈值（天）
            test_size: 测试集比例
        """
        print(f"📊 准备数据...")
        print(f"流失定义: Recency > {churn_threshold} 天")
        
        # 创建流失标签
        self.rfm['Churn'] = (self.rfm['Recency'] > churn_threshold).astype(int)
        
        # 特征选择
        features = ['Recency', 'Frequency', 'Monetary']
        X = self.rfm[features]
        y = self.rfm['Churn']
        
        print(f"总样本数: {len(X)}")
        print(f"流失用户: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"活跃用户: {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(self.X_train)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        
        return self
    
    def train_logistic_regression(self):
        """训练逻辑回归模型"""
        print("\n🤖 训练逻辑回归模型...")
        
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        
        # 预测
        self.predictions = self.model.predict(self.X_test)
        self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
        
        # 评估
        accuracy = accuracy_score(self.y_test, self.predictions)
        print(f"模型准确率: {accuracy:.4f}")
        
        return self
    
    def train_random_forest(self):
        """训练随机森林模型（对比用）"""
        print("\n🌲 训练随机森林模型...")
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        
        rf_predictions = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        
        print(f"随机森林准确率: {rf_accuracy:.4f}")
        
        # 特征重要性
        importances = pd.DataFrame({
            'feature': ['Recency', 'Frequency', 'Monetary'],
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性:")
        print(importances)
        
        return rf_model
    
    def evaluate_model(self):
        """详细评估模型性能"""
        print("\n📈 模型评估报告:")
        print("="*50)
        print(classification_report(self.y_test, self.predictions, 
                                   target_names=['活跃', '流失']))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.predictions)
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
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, save_path='../data/confusion_matrix.png'):
        """绘制混淆矩阵热力图"""
        cm = confusion_matrix(self.y_test, self.predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['活跃', '流失'],
                   yticklabels=['活跃', '流失'])
        
        ax.set_title('流失预测 - 混淆矩阵', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, save_path='../data/roc_curve.png'):
        """绘制 ROC 曲线"""
        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='随机分类器')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        ax.set_title('流失预测 - ROC 曲线', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC 曲线已保存: {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_feature_importance(self, save_path='../data/feature_importance.png'):
        """绘制特征重要性（逻辑回归系数）"""
        # 对于逻辑回归，系数代表特征重要性
        coefficients = self.model.coef_[0]
        features = ['Recency', 'Frequency', 'Monetary']
        
        # 归一化系数到 0-1 范围
        importance = np.abs(coefficients)
        importance = importance / importance.sum()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax.bar(features, importance, color=colors, edgecolor='black')
        
        # 添加数值标签
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title('特征重要性分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('特征', fontsize=12)
        ax.set_ylabel('重要性权重', fontsize=12)
        ax.set_ylim(0, max(importance) * 1.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 特征重要性图已保存: {save_path}")
        plt.close()
    
    def predict_churn_risk(self, output_path='../data/churn_predictions.csv'):
        """
        为所有用户预测流失风险
        
        Returns:
            DataFrame 包含预测结果
        """
        # 为所有用户预测
        all_features = self.rfm[['Recency', 'Frequency', 'Monetary']]
        self.rfm['Churn_Probability'] = self.model.predict_proba(all_features)[:, 1]
        self.rfm['Churn_Risk_Level'] = pd.cut(
            self.rfm['Churn_Probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['低风险', '中风险', '高风险']
        )
        
        # 保存预测结果
        self.rfm.to_csv(output_path)
        print(f"\n✅ 流失预测结果已保存: {output_path}")
        
        # 统计风险分布
        risk_dist = self.rfm['Churn_Risk_Level'].value_counts()
        print("\n流失风险分布:")
        for level, count in risk_dist.items():
            pct = count / len(self.rfm) * 100
            print(f"  {level}: {count} 人 ({pct:.1f}%)")
        
        return self.rfm
    
    def save_model(self, path='../data/churn_model.pkl'):
        """保存训练好的模型"""
        joblib.dump(self.model, path)
        print(f"\n✅ 模型已保存: {path}")


def main():
    """主函数"""
    import os
    
    # 加载 RFM 数据
    print("="*60)
    print("🚀 用户流失预测分析")
    print("="*60)
    
    rfm_path = '../data/rfm_results.csv'
    if not os.path.exists(rfm_path):
        print(f"错误: 找不到 RFM 数据文件 {rfm_path}")
        print("请先运行 rfm_analysis.py")
        return
    
    rfm_df = pd.read_csv(rfm_path, index_col=0)
    
    # 创建预测器
    predictor = ChurnPredictor(rfm_df)
    
    # 准备数据
    predictor.prepare_data(churn_threshold=60)
    
    # 训练模型
    predictor.train_logistic_regression()
    
    # 对比随机森林
    predictor.train_random_forest()
    
    # 评估
    predictor.evaluate_model()
    
    # 生成可视化
    print("\n" + "="*60)
    print("🎨 生成可视化图表...")
    print("="*60)
    os.makedirs('../data', exist_ok=True)
    
    predictor.plot_confusion_matrix()
    predictor.plot_roc_curve()
    predictor.plot_feature_importance()
    
    # 预测所有用户
    predictor.predict_churn_risk()
    
    # 保存模型
    predictor.save_model()
    
    print("\n" + "="*60)
    print("✅ 流失预测分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()
