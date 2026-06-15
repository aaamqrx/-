# -*- coding: utf-8 -*-
"""
数据探索性分析(EDA)与可视化脚本

功能：
1. 绘制评分与评价人数的散点图
2. 分析不同档期的电影表现差异（柱状图）
3. 绘制上映前百度指数与最终评价人数的相关性分析
4. 生成统计摘要报告
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from scipy import stats


def configure_chinese_font():
    """配置中文字体支持，返回字体属性对象"""
    import os
    from pathlib import Path

    # Windows 系统字体目录
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path(os.environ.get('WINDIR', 'C:/Windows')) / "Fonts"
    ]

    # 按优先级查找字体文件
    font_files = [
        ('msyh.ttc', 'Microsoft YaHei'),      # 微软雅黑
        ('msyhbd.ttc', 'Microsoft YaHei'),    # 微软雅黑粗体
        ('simhei.ttf', 'SimHei'),             # 黑体
        ('simsun.ttc', 'SimSun'),             # 宋体
        ('simkai.ttf', 'KaiTi'),              # 楷体
    ]

    selected_font = None
    font_path = None

    # 查找可用字体文件
    for font_file, font_name in font_files:
        for font_dir in font_dirs:
            test_path = font_dir / font_file
            if test_path.exists():
                font_path = str(test_path)
                selected_font = font_name
                break
        if font_path:
            break

    if font_path:
        # 使用找到的字体文件
        from matplotlib.font_manager import FontProperties
        font_prop = FontProperties(fname=font_path)

        # 设置全局字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False

        print(f"✓ 已配置中文字体: {selected_font}")
        print(f"  字体路径: {font_path}")

        return font_prop  # 返回字体属性对象
    else:
        # 降级方案
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"⚠ 未找到字体文件，使用系统默认配置")

        return None

    # 强制重新加载字体管理器
    import matplotlib
    matplotlib.font_manager._load_fontmanager(try_read_cache=False)


def load_and_clean_data():
    """加载并清洗数据"""
    print("\n" + "="*60)
    print("步骤 1: 加载数据")
    print("="*60)

    data_file = config.FINAL_DIR / "china_all.xlsx"

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    print(f"正在读取: {data_file}")
    df = pd.read_excel(data_file)

    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()[:10]}...")

    # 数据清洗
    print("\n清洗数据...")

    # 创建副本
    df_clean = df.copy()

    # 基础字段清洗
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['评分', '评价人数'])
    print(f"  - 移除评分或评价人数缺失的记录: {initial_rows - len(df_clean)} 条")

    # 确保数值类型
    df_clean['评分'] = pd.to_numeric(df_clean['评分'], errors='coerce')
    df_clean['评价人数'] = pd.to_numeric(df_clean['评价人数'], errors='coerce')

    # 移除异常值（评分应在0-10之间，评价人数应为正数）
    before_filter = len(df_clean)
    df_clean = df_clean[(df_clean['评分'] >= 0) & (df_clean['评分'] <= 10)]
    df_clean = df_clean[df_clean['评价人数'] > 0]
    print(f"  - 移除异常值: {before_filter - len(df_clean)} 条")

    # 处理百度指数字段
    if '上映前7天百度搜索指数均值' in df_clean.columns:
        df_clean['上映前7天百度搜索指数均值'] = pd.to_numeric(
            df_clean['上映前7天百度搜索指数均值'], errors='coerce'
        )

    print(f"\n清洗后数据形状: {df_clean.shape}")
    print(f"清洗后保留: {len(df_clean)}/{initial_rows} ({len(df_clean)/initial_rows*100:.1f}%)")

    return df_clean


def plot_rating_vs_reviews(df, font_prop=None):
    """绘制评分与评价人数的散点图"""
    print("\n" + "="*60)
    print("步骤 2: 绘制评分与评价人数散点图")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: 线性坐标
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['评价人数'], df['评分'],
                          alpha=0.5, s=30, c=df['评分'],
                          cmap='viridis', edgecolors='none')
    ax1.set_xlabel('评价人数', fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel('评分', fontsize=12, fontproperties=font_prop)
    ax1.set_title('电影评分与评价人数关系图 (线性坐标)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('评分', fontproperties=font_prop)

    # 图2: 对数坐标（评价人数）
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['评价人数'], df['评分'],
                          alpha=0.5, s=30, c=df['评分'],
                          cmap='viridis', edgecolors='none')
    ax2.set_xscale('log')
    ax2.set_xlabel('评价人数 (对数尺度)', fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel('评分', fontsize=12, fontproperties=font_prop)
    ax2.set_title('电影评分与评价人数关系图 (对数坐标)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('评分', fontproperties=font_prop)

    # 计算相关系数
    corr_pearson = df['评分'].corr(df['评价人数'])
    corr_spearman = df['评分'].corr(df['评价人数'], method='spearman')

    # 添加统计信息
    stats_text = f'样本数: {len(df)}\nPearson相关系数: {corr_pearson:.3f}\nSpearman相关系数: {corr_spearman:.3f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontproperties=font_prop)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_file = config.EDA_OUTPUT_DIR / "scatter_rating_vs_reviews.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")

    plt.close()

    print(f"  Pearson 相关系数: {corr_pearson:.4f}")
    print(f"  Spearman 相关系数: {corr_spearman:.4f}")


def plot_schedule_comparison(df, font_prop=None):
    """绘制不同档期电影表现对比柱状图"""
    print("\n" + "="*60)
    print("步骤 3: 绘制不同档期电影表现对比")
    print("="*60)

    # 识别档期列
    schedule_cols = ['是否春节档', '是否国庆档', '是否五一档', '是否暑期档', '是否普通周末']
    available_cols = [col for col in schedule_cols if col in df.columns]

    if not available_cols:
        print("⚠ 警告: 未找到档期字段，跳过此图表")
        return

    # 准备数据
    schedule_data = []
    for col in available_cols:
        schedule_name = col.replace('是否', '')
        mask = df[col] == 1
        if mask.sum() > 0:
            schedule_data.append({
                '档期': schedule_name,
                '平均评分': df[mask]['评分'].mean(),
                '评分标准差': df[mask]['评分'].std(),
                '平均评价人数': df[mask]['评价人数'].mean(),
                '评价人数中位数': df[mask]['评价人数'].median(),
                '样本数': mask.sum()
            })

    schedule_df = pd.DataFrame(schedule_data)

    if schedule_df.empty:
        print("⚠ 警告: 没有档期数据，跳过此图表")
        return

    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 图1: 平均评分对比
    ax1 = axes[0]
    bars1 = ax1.bar(schedule_df['档期'], schedule_df['平均评分'],
                    color='steelblue', alpha=0.7, edgecolor='black')
    ax1.errorbar(schedule_df['档期'], schedule_df['平均评分'],
                yerr=schedule_df['评分标准差'], fmt='none', ecolor='red',
                capsize=5, alpha=0.6, label='标准差')
    ax1.set_ylabel('平均评分', fontsize=12, fontproperties=font_prop)
    ax1.set_title('不同档期电影平均评分对比', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(prop=font_prop)

    # 设置x轴标签字体
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_prop)

    # 在柱子上添加数值和样本数
    for i, (bar, row) in enumerate(zip(bars1, schedule_df.itertuples())):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}\n(n={row.样本数})',
                ha='center', va='bottom', fontsize=9, fontproperties=font_prop)

    # 图2: 评价人数对比
    ax2 = axes[1]
    bars2 = ax2.bar(schedule_df['档期'], schedule_df['评价人数中位数'],
                    color='coral', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('评价人数中位数', fontsize=12, fontproperties=font_prop)
    ax2.set_xlabel('档期', fontsize=12, fontproperties=font_prop)
    ax2.set_title('不同档期电影评价人数中位数对比', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax2.grid(axis='y', alpha=0.3)

    # 设置x轴标签字体
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)

    # 在柱子上添加数值
    for bar, row in zip(bars2, schedule_df.itertuples()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontproperties=font_prop)

    plt.tight_layout()

    output_file = config.EDA_OUTPUT_DIR / "bar_schedule_comparison.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")

    plt.close()

    # 输出统计摘要
    print("\n档期统计摘要:")
    print(schedule_df.to_string(index=False))


def plot_baidu_correlation(df, font_prop=None):
    """绘制上映前百度指数与评价人数的相关性分析"""
    print("\n" + "="*60)
    print("步骤 4: 绘制百度指数与评价人数相关性")
    print("="*60)

    baidu_col = '上映前7天百度搜索指数均值'

    if baidu_col not in df.columns:
        print(f"⚠ 警告: 未找到列 '{baidu_col}'，跳过此图表")
        return

    # 筛选有效百度指数数据
    df_baidu = df[df[baidu_col].notna() & (df[baidu_col] > 0)].copy()

    if len(df_baidu) < 10:
        print(f"⚠ 警告: 有效百度指数数据不足 ({len(df_baidu)} 条)，跳过此图表")
        return

    print(f"有效百度指数样本数: {len(df_baidu)}/{len(df)} ({len(df_baidu)/len(df)*100:.1f}%)")

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: 散点图 + 回归线
    ax1 = axes[0]
    scatter = ax1.scatter(df_baidu[baidu_col], df_baidu['评价人数'],
                         alpha=0.5, s=30, c=df_baidu['评分'],
                         cmap='coolwarm', edgecolors='none')
    ax1.set_xlabel('上映前7天百度搜索指数均值', fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel('评价人数', fontsize=12, fontproperties=font_prop)
    ax1.set_title('百度指数与评价人数关系 (线性)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('评分', fontproperties=font_prop)

    # 添加回归线
    z = np.polyfit(df_baidu[baidu_col], df_baidu['评价人数'], 1)
    p = np.poly1d(z)
    ax1.plot(df_baidu[baidu_col], p(df_baidu[baidu_col]), "r--", alpha=0.8, linewidth=2, label='线性拟合')
    ax1.legend(prop=font_prop)

    # 图2: 对数坐标散点图
    ax2 = axes[1]
    scatter2 = ax2.scatter(df_baidu[baidu_col], df_baidu['评价人数'],
                          alpha=0.5, s=30, c=df_baidu['评分'],
                          cmap='coolwarm', edgecolors='none')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('上映前7天百度搜索指数均值 (对数)', fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel('评价人数 (对数)', fontsize=12, fontproperties=font_prop)
    ax2.set_title('百度指数与评价人数关系 (对数)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('评分', fontproperties=font_prop)

    # 计算相关系数
    corr_pearson = df_baidu[baidu_col].corr(df_baidu['评价人数'])
    corr_spearman = df_baidu[baidu_col].corr(df_baidu['评价人数'], method='spearman')

    # 添加统计信息
    stats_text = (f'样本数: {len(df_baidu)}\n'
                 f'Pearson相关系数: {corr_pearson:.3f}\n'
                 f'Spearman相关系数: {corr_spearman:.3f}')
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             fontproperties=font_prop)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_file = config.EDA_OUTPUT_DIR / "correlation_baidu_vs_reviews.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")

    plt.close()

    print(f"  Pearson 相关系数: {corr_pearson:.4f}")
    print(f"  Spearman 相关系数: {corr_spearman:.4f}")

    # 简单的预测能力评估
    if corr_pearson > 0.3:
        print("  → 百度指数对评价人数有一定预测能力")
    else:
        print("  → 百度指数对评价人数的预测能力较弱")


def generate_summary_statistics(df):
    """生成统计摘要报告"""
    print("\n" + "="*60)
    print("步骤 5: 生成统计摘要")
    print("="*60)

    summary = {
        '总样本数': len(df),
        '评分_均值': df['评分'].mean(),
        '评分_中位数': df['评分'].median(),
        '评分_标准差': df['评分'].std(),
        '评价人数_均值': df['评价人数'].mean(),
        '评价人数_中位数': df['评价人数'].median(),
        '评价人数_标准差': df['评价人数'].std(),
    }

    # 百度指数统计
    baidu_col = '上映前7天百度搜索指数均值'
    if baidu_col in df.columns:
        valid_baidu = df[df[baidu_col].notna()]
        summary['百度指数_有效样本数'] = len(valid_baidu)
        summary['百度指数_均值'] = valid_baidu[baidu_col].mean()
        summary['百度指数_中位数'] = valid_baidu[baidu_col].median()

    # 输出到控制台
    print("\n数据集统计摘要:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 保存到CSV
    summary_df = pd.DataFrame([summary])
    output_file = config.EDA_OUTPUT_DIR / "summary_statistics.csv"
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 统计摘要已保存: {output_file}")

    return summary


def main():
    """主函数"""
    print("\n" + "="*60)
    print("豆瓣电影数据探索性分析 (EDA) 与可视化")
    print("="*60)

    # 配置环境，获取字体属性
    font_prop = configure_chinese_font()

    # 设置绘图样式
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        plt.style.use('seaborn-v0_8-whitegrid')

    # 全局设置，确保所有图表都使用相同配置
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.dpi'] = config.FIGURE_DPI
    plt.rcParams['font.size'] = config.PLOT_FONT_SIZE

    # 1. 加载数据
    df = load_and_clean_data()

    # 2. 生成可视化（传递字体属性）
    plot_rating_vs_reviews(df, font_prop)
    plot_schedule_comparison(df, font_prop)
    plot_baidu_correlation(df, font_prop)

    # 3. 生成统计摘要
    generate_summary_statistics(df)

    print("\n" + "="*60)
    print("✓ 所有分析完成！")
    print(f"输出目录: {config.EDA_OUTPUT_DIR}")
    print("\n生成的文件:")
    print("  1. scatter_rating_vs_reviews.png - 评分与评价人数散点图")
    print("  2. bar_schedule_comparison.png - 档期对比柱状图")
    print("  3. correlation_baidu_vs_reviews.png - 百度指数相关性")
    print("  4. summary_statistics.csv - 统计摘要")
    print("="*60)


if __name__ == "__main__":
    main()