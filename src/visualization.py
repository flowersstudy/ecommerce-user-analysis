"""
可视化模�?生成各类分析图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_user_distribution(rfm_df, save_path='../data/user_distribution.png'):
    """
    绘制用户分群分布�?    
    Args:
        rfm_df: 包含 Segment 列的 RFM DataFrame
        save_path: 保存路径
    """
    segment_counts = rfm_df['Segment'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(segment_counts.index, segment_counts.values, color=colors)
    
    # 添加数值标�?    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    ax.set_title('用户分群分布', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('用户类型', fontsize=12)
    ax.set_ylabel('用户数量', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def plot_rfm_heatmap(rfm_df, save_path='../data/rfm_heatmap.png'):
    """
    绘制 RFM 热力�?    
    展示 Frequency �?Monetary 在不�?Recency 区间的分�?    """
    # 创建 Recency 分组
    rfm_df = rfm_df.copy()
    rfm_df['R_Group'] = pd.cut(rfm_df['Recency'], 
                                bins=[0, 30, 60, 90, 180, 400],
                                labels=['0-30�?, '31-60�?, '61-90�?, '91-180�?, '180�?'])
    
    # 创建透视�?    pivot_table = rfm_df.groupby('R_Group').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Frequency 热力�?    sns.heatmap(pivot_table[['Frequency']], annot=True, fmt='.1f', 
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': '平均购买频次'})
    axes[0].set_title('各活跃区间的平均购买频次', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('最近购买时�?, fontsize=12)
    
    # Monetary 热力�?    sns.heatmap(pivot_table[['Monetary']], annot=True, fmt='.0f',
                cmap='YlGn', ax=axes[1], cbar_kws={'label': '平均消费金额'})
    axes[1].set_title('各活跃区间的平均消费金额', fontsize=14)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def plot_monetary_distribution(rfm_df, save_path='../data/monetary_distribution.png'):
    """绘制各群体的消费金额分布箱型�?""
    
    segments = ['高价值用�?, '流失风险用户', '忠实用户', '新用�?]
    data_to_plot = [rfm_df[rfm_df['Segment'] == seg]['Monetary'].values 
                    for seg in segments]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data_to_plot, labels=segments, patch_artist=True,
                    notch=True, showfliers=False)
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('各用户群体的消费金额分布', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('用户类型', fontsize=12)
    ax.set_ylabel('消费金额 (£)', fontsize=12)
    ax.set_ylim(0, rfm_df['Monetary'].quantile(0.95))  # 限制 y 轴范围去除极端�?    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def plot_frequency_distribution(rfm_df, save_path='../data/frequency_distribution.png'):
    """绘制各群体的购买频次分布"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    segments = ['高价值用�?, '流失风险用户', '忠实用户', '新用�?]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    for seg, color in zip(segments, colors):
        data = rfm_df[rfm_df['Segment'] == seg]['Frequency']
        ax.hist(data, bins=20, alpha=0.6, label=seg, color=color, edgecolor='black')
    
    ax.set_title('各用户群体的购买频次分布', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('购买频次', fontsize=12)
    ax.set_ylabel('用户数量', fontsize=12)
    ax.legend()
    ax.set_xlim(0, rfm_df['Frequency'].quantile(0.95))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def plot_clustering_scatter(rfm_df, save_path='../data/user_clustering.png'):
    """
    绘制用户聚类散点�?    
    需�?rfm_df 包含 Cluster �?    """
    from sklearn.cluster import KMeans
    
    # 如果没有 Cluster 列，先进行聚�?    if 'Cluster' not in rfm_df.columns:
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm_df = rfm_df.copy()
        rfm_df['Cluster'] = kmeans.fit_predict(
            rfm_df[['Recency', 'Frequency', 'Monetary']]
        )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # �?1: Frequency vs Monetary
    scatter1 = axes[0].scatter(
        rfm_df['Frequency'], 
        rfm_df['Monetary'],
        c=rfm_df['Cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=50
    )
    axes[0].set_xlabel('购买频次', fontsize=12)
    axes[0].set_ylabel('消费金额', fontsize=12)
    axes[0].set_title('KMeans 聚类: 频次 vs 金额', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, rfm_df['Frequency'].quantile(0.95))
    axes[0].set_ylim(0, rfm_df['Monetary'].quantile(0.95))
    plt.colorbar(scatter1, ax=axes[0], label='聚类')
    
    # �?2: Recency vs Monetary
    scatter2 = axes[1].scatter(
        rfm_df['Recency'], 
        rfm_df['Monetary'],
        c=rfm_df['Cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=50
    )
    axes[1].set_xlabel('最近购买天�?(Recency)', fontsize=12)
    axes[1].set_ylabel('消费金额', fontsize=12)
    axes[1].set_title('KMeans 聚类: 活跃�?vs 金额', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, rfm_df['Monetary'].quantile(0.95))
    plt.colorbar(scatter2, ax=axes[1], label='聚类')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def plot_segment_comparison(rfm_df, save_path='../data/segment_comparison.png'):
    """绘制各群体的 RFM 均值对比雷达图"""
    from math import pi
    
    # 计算各群体的平均 RFM
    segments = ['高价值用�?, '流失风险用户', '忠实用户', '新用�?]
    
    # 归一化数据用于雷达图
    rfm_norm = rfm_df.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        rfm_norm[col] = (rfm_df[col] - rfm_df[col].min()) / (rfm_df[col].max() - rfm_df[col].min())
    
    # Recency 需要反向（越小越好�?    rfm_norm['Recency'] = 1 - rfm_norm['Recency']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Recency\n(活跃�?', 'Frequency\n(频次)', 'Monetary\n(金额)']
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    for seg, color in zip(segments, colors):
        values = rfm_norm[rfm_norm['Segment'] == seg][['Recency', 'Frequency', 'Monetary']].mean().values
        values = values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=seg, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('用户群体 RFM 特征对比', fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"�?图表已保�? {save_path}")
    plt.close()


def generate_all_plots(rfm_df, output_dir='../outputs'):
    """
    生成所有可视化图表
    
    Args:
        rfm_df: RFM DataFrame
        output_dir: 输出目录
    """
    import pandas as pd
    
    print("\n🎨 正在生成可视化图�?..")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 用户分布
    plot_user_distribution(rfm_df, f'{output_dir}/user_distribution.png')
    
    # 2. RFM 热力�?    plot_rfm_heatmap(rfm_df, f'{output_dir}/rfm_heatmap.png')
    
    # 3. 消费金额分布
    plot_monetary_distribution(rfm_df, f'{output_dir}/monetary_distribution.png')
    
    # 4. 购买频次分布
    plot_frequency_distribution(rfm_df, f'{output_dir}/frequency_distribution.png')
    
    # 5. 聚类散点�?    plot_clustering_scatter(rfm_df, f'{output_dir}/user_clustering.png')
    
    # 6. 雷达图对�?    plot_segment_comparison(rfm_df, f'{output_dir}/segment_comparison.png')
    
    print(f"\n�?所有图表已保存�?{output_dir}/")


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    rfm_df = pd.read_csv('../data/rfm_results.csv', index_col=0)
    generate_all_plots(rfm_df)
