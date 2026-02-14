"""
RFM 分析模块
实现用户价值分群和 RFM 评分计算
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class RFMAnalyzer:
    """RFM 分析器类"""
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        Args:
            data_path: CSV 数据文件路径
        """
        self.data = None
        self.rfm = None
        self.data_path = data_path
        
    def load_data(self):
        """加载并清洗数据"""
        print("📊 正在加载数据...")
        
        # 加载数据
        df = pd.read_csv(self.data_path, encoding='latin1')
        print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 数据清洗
        # 1. 删除没有用户ID的行
        df = df.dropna(subset=['CustomerID'])
        
        # 2. 删除退货记录
        df = df[df['Quantity'] > 0]
        
        # 3. 转换时间格式
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # 4. 计算金额
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        self.data = df
        print(f"清洗后数据: {df.shape[0]} 行, 用户数: {df['CustomerID'].nunique()}")
        
        return self
    
    def calculate_rfm(self, n_quantiles=5):
        """
        计算 RFM 指标
        
        Args:
            n_quantiles: 分位数数量 (默认 5 分位)
        """
        print("\n📈 正在计算 RFM 指标...")
        
        # 获取数据最后一天作为快照日期
        snapshot_date = self.data['InvoiceDate'].max()
        print(f"数据快照日期: {snapshot_date}")
        
        # 按用户分组计算 RFM
        rfm = self.data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # R
            'InvoiceNo': 'nunique',                                   # F
            'TotalPrice': 'sum'                                       # M
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # 计算 RFM 评分 (1-n_quantiles)
        rfm['R_Score'] = pd.qcut(
            rfm['Recency'], 
            n_quantiles, 
            labels=list(range(n_quantiles, 0, -1)),  # R 越小越好
            duplicates='drop'
        )
        
        rfm['F_Score'] = pd.qcut(
            rfm['Frequency'].rank(method='first'), 
            n_quantiles, 
            labels=list(range(1, n_quantiles + 1)),  # F 越大越好
            duplicates='drop'
        )
        
        rfm['M_Score'] = pd.qcut(
            rfm['Monetary'], 
            n_quantiles, 
            labels=list(range(1, n_quantiles + 1)),  # M 越大越好
            duplicates='drop'
        )
        
        # 组合 RFM 分数
        rfm['RFM_Score'] = (
            rfm['R_Score'].astype(str) + 
            rfm['F_Score'].astype(str) + 
            rfm['M_Score'].astype(str)
        )
        
        self.rfm = rfm
        print(f"RFM 计算完成: {rfm.shape[0]} 个用户")
        
        return self
    
    def segment_users(self):
        """用户分群"""
        print("\n🎯 正在进行用户分群...")
        
        # 定义分群规则
        def get_segment(row):
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            
            # 高价值用户: R,F,M 都很高
            if r >= 4 and f >= 4 and m >= 4:
                return '高价值用户'
            
            # 流失风险用户: R,F,M 都很低
            elif r <= 2 and f <= 2 and m <= 2:
                return '流失风险用户'
            
            # 忠实用户: F,M 高但 R 不一定高
            elif f >= 4 and m >= 4:
                return '忠实用户'
            
            # 新用户: R 高但 F,M 低
            elif r >= 4 and f <= 2:
                return '新用户'
            
            else:
                return '其他用户'
        
        self.rfm['Segment'] = self.rfm.apply(get_segment, axis=1)
        
        # 统计各群体数量
        segment_counts = self.rfm['Segment'].value_counts()
        print("\n用户分群结果:")
        for segment, count in segment_counts.items():
            pct = count / len(self.rfm) * 100
            print(f"  {segment}: {count} 人 ({pct:.1f}%)")
        
        return self
    
    def get_segment_stats(self):
        """获取各群体统计信息"""
        stats = self.rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            'CustomerID': 'count'
        }).round(2)
        
        stats.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Monetary', 'Count']
        stats['Percentage'] = (stats['Count'] / stats['Count'].sum() * 100).round(1)
        
        return stats.sort_values('Total_Monetary', ascending=False)
    
    def save_results(self, output_path):
        """保存 RFM 结果到 CSV"""
        self.rfm.to_csv(output_path)
        print(f"\n✅ RFM 结果已保存: {output_path}")


def main():
    """主函数"""
    # 初始化分析器
    analyzer = RFMAnalyzer('../data_analysis/data.csv')
    
    # 执行分析流程
    analyzer.load_data()\
           .calculate_rfm(n_quantiles=5)\
           .segment_users()
    
    # 输出统计信息
    print("\n" + "="*50)
    print("📊 各群体详细统计:")
    print("="*50)
    stats = analyzer.get_segment_stats()
    print(stats)
    
    # 保存结果
    analyzer.save_results('../outputs/rfm_results.csv')
    
    return analyzer


if __name__ == '__main__':
    analyzer = main()
