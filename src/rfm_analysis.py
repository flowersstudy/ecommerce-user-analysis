"""
RFM 分析模块
实现用户价值分群和 RFM 评分计算
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_rfm_analysis():
    """运行 RFM 分析的主要函数"""
    print("📊 正在加载数据...")
    
    # 计算项目根目录路径
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到数据文件 {DATA_PATH}")
        print("请确保 data.csv 在 data/ 目录下")
        return
    
    # 加载数据
    df = pd.read_csv(DATA_PATH, encoding='latin1')
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
    
    print(f"清洗后数据: {df.shape[0]} 行, 用户数: {df['CustomerID'].nunique()}")
    
    # 获取数据最后一天作为快照日期
    snapshot_date = df['InvoiceDate'].max()
    print(f"数据快照日期: {snapshot_date}")
    
    # 按用户分组计算 RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # R
        'InvoiceNo': 'nunique',                                   # F
        'TotalPrice': 'sum'                                       # M
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # 计算 RFM 评分 (1-5分)
    rfm['R_Score'] = pd.qcut(
        rfm['Recency'], 
        5, 
        labels=[5, 4, 3, 2, 1],  # R 越小越好
        duplicates='drop'
    )
    
    rfm['F_Score'] = pd.qcut(
        rfm['Frequency'].rank(method='first'), 
        5, 
        labels=[1, 2, 3, 4, 5],  # F 越大越好
        duplicates='drop'
    )
    
    rfm['M_Score'] = pd.qcut(
        rfm['Monetary'], 
        5, 
        labels=[1, 2, 3, 4, 5],  # M 越大越好
        duplicates='drop'
    )
    
    # 组合 RFM 分数
    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) + 
        rfm['F_Score'].astype(str) + 
        rfm['M_Score'].astype(str)
    )
    
    print(f"RFM 计算完成: {rfm.shape[0]} 个用户")
    
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
    
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    
    # 统计各群体数量
    segment_counts = rfm['Segment'].value_counts()
    print("\n用户分群结果:")
    for segment, count in segment_counts.items():
        pct = count / len(rfm) * 100
        print(f"  {segment}: {count} 人 ({pct:.1f}%)")
    
    # 保存结果
    OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "rfm_results.csv")
    rfm.to_csv(output_path)
    print(f"\n✅ RFM 结果已保存: {output_path}")
    
    print("RFM done")


def main():
    """主函数"""
    run_rfm_analysis()


if __name__ == '__main__':
    main()
