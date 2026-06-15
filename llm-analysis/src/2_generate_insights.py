# -*- coding: utf-8 -*-
"""
步骤2: 生成可视化和洞察

功能:
1. 加载打标签的电影数据
2. 生成4个主要可视化
3. 输出统计分析
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import configure_chinese_font

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx


def load_tagged_data():
    """加载打好标签的数据"""
    print("\n" + "="*60)
    print("步骤 2.1: 加载数据")
    print("="*60)

    data_file = config.OUTPUT_DIR / "tagged_movies.csv"
    if not data_file.exists():
        raise FileNotFoundError(
            f"未找到标签数据文件: {data_file}\n"
            f"请先运行: python src/1_analyze_plots.py"
        )

    df = pd.read_csv(data_file)
    print(f"加载数据: {data_file}")
    print(f"总记录数: {len(df)}")

    # 筛选有标签的记录
    df_tagged = df[df['LLM标签'].notna() & (df['LLM标签'] != '')].copy()
    print(f"有标签记录: {len(df_tagged)}")

    # 解析标签列表
    df_tagged['标签列表'] = df_tagged['LLM标签'].apply(
        lambda x: [t.strip() for t in x.split(',') if t.strip()]
    )

    return df_tagged


def generate_tag_distribution(df_tagged, font_prop, top_n=20):
    """生成标签频率分布图

    Args:
        df_tagged: 标签数据 DataFrame
        font_prop: 中文字体属性
        top_n: 显示前N个标签
    """
    print("\n" + "="*60)
    print("步骤 2.2: 生成标签分布图")
    print("="*60)

    # 统计所有标签
    all_tags = []
    for tags in df_tagged['标签列表']:
        all_tags.extend(tags)

    tag_counts = Counter(all_tags)
    top_tags = tag_counts.most_common(top_n)

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    tags, counts = zip(*top_tags)
    percentages = [count / len(df_tagged) * 100 for count in counts]

    y_pos = np.arange(len(tags))
    bars = ax.barh(y_pos, percentages, color=sns.color_palette("viridis", len(tags)))

    # 添加数值标签
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        ax.text(pct + 0.5, i, f'{count}次 ({pct:.1f}%)',
                va='center', fontproperties=font_prop, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontproperties=font_prop, fontsize=10)
    ax.set_xlabel('电影占比 (%)', fontproperties=font_prop, fontsize=11)
    ax.set_title(f'Top {top_n} 电影标签分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    output_file = config.VIZ_DIR / "tag_distribution.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存: {output_file}")
    print(f"  Top 5 标签: {', '.join([f'{t}({c})' for t, c in top_tags[:5]])}")


def generate_schedule_tag_heatmap(df_tagged, font_prop, top_n=15):
    """生成档期-标签热力图

    Args:
        df_tagged: 标签数据 DataFrame
        font_prop: 中文字体属性
        top_n: 显示前N个标签
    """
    print("\n" + "="*60)
    print("步骤 2.3: 生成档期-标签热力图")
    print("="*60)

    # 定义档期列
    schedule_cols = ['是否春节档', '是否国庆档', '是否五一档', '是否暑期档', '是否普通周末']
    schedule_names = ['春节档', '国庆档', '五一档', '暑期档', '普通周末']

    # 获取top标签
    all_tags = []
    for tags in df_tagged['标签列表']:
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)
    top_tags = [tag for tag, _ in tag_counts.most_common(top_n)]

    # 构建热力图数据
    heatmap_data = []
    for schedule_col, schedule_name in zip(schedule_cols, schedule_names):
        if schedule_col not in df_tagged.columns:
            continue

        schedule_movies = df_tagged[df_tagged[schedule_col] == 1]
        if len(schedule_movies) == 0:
            continue

        row_data = []
        for tag in top_tags:
            tag_count = sum(tag in tags for tags in schedule_movies['标签列表'])
            percentage = tag_count / len(schedule_movies) * 100
            row_data.append(percentage)

        heatmap_data.append(row_data)

    # 创建热力图
    if heatmap_data:
        fig, ax = plt.subplots(figsize=(12, 6))

        heatmap_df = pd.DataFrame(heatmap_data,
                                   index=[name for name, col in zip(schedule_names, schedule_cols)
                                          if col in df_tagged.columns],
                                   columns=top_tags)

        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': '电影占比 (%)'}, ax=ax,
                    linewidths=0.5, linecolor='white')

        ax.set_xlabel('标签', fontproperties=font_prop, fontsize=11)
        ax.set_ylabel('档期', fontproperties=font_prop, fontsize=11)
        ax.set_title('不同档期的标签分布热力图', fontproperties=font_prop, fontsize=14, fontweight='bold')

        # 设置刻度标签字体
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop, rotation=0)

        plt.tight_layout()
        output_file = config.VIZ_DIR / "schedule_tag_heatmap.png"
        plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ 保存: {output_file}")


def generate_tag_performance(df_tagged, font_prop, top_n=20):
    """生成标签性能分析图

    Args:
        df_tagged: 标签数据 DataFrame
        font_prop: 中文字体属性
        top_n: 显示前N个标签
    """
    print("\n" + "="*60)
    print("步骤 2.4: 生成标签性能分析图")
    print("="*60)

    # 获取top标签
    all_tags = []
    for tags in df_tagged['标签列表']:
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)
    top_tags = [tag for tag, count in tag_counts.most_common(top_n) if count >= 5]  # 至少5部电影

    # 计算每个标签的平均评分和评价人数
    tag_stats = []
    for tag in top_tags:
        tag_movies = df_tagged[df_tagged['标签列表'].apply(lambda tags: tag in tags)]

        if len(tag_movies) > 0 and '评分' in tag_movies.columns and '评价人数' in tag_movies.columns:
            avg_rating = tag_movies['评分'].mean()
            avg_reviews = tag_movies['评价人数'].mean()
            count = len(tag_movies)
            tag_stats.append({
                'tag': tag,
                'avg_rating': avg_rating,
                'avg_reviews': avg_reviews,
                'count': count
            })

    if not tag_stats:
        print("⚠️  无法生成性能分析图：缺少评分或评价人数数据")
        return

    # 创建散点图
    fig, ax = plt.subplots(figsize=(12, 8))

    df_stats = pd.DataFrame(tag_stats)

    scatter = ax.scatter(df_stats['avg_rating'], df_stats['avg_reviews'],
                         s=df_stats['count'] * 3, alpha=0.6,
                         c=range(len(df_stats)), cmap='viridis')

    # 添加标签
    for _, row in df_stats.iterrows():
        ax.annotate(row['tag'], (row['avg_rating'], row['avg_reviews']),
                    fontproperties=font_prop, fontsize=9, alpha=0.7)

    ax.set_xlabel('平均评分', fontproperties=font_prop, fontsize=11)
    ax.set_ylabel('平均评价人数 (对数刻度)', fontproperties=font_prop, fontsize=11)
    ax.set_yscale('log')
    ax.set_title('标签性能分析 (气泡大小=电影数量)', fontproperties=font_prop, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = config.VIZ_DIR / "tag_performance.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存: {output_file}")


def generate_tag_cooccurrence(df_tagged, font_prop, top_n=15, min_cooccur=5):
    """生成标签共现网络图

    Args:
        df_tagged: 标签数据 DataFrame
        font_prop: 中文字体属性
        top_n: 显示前N个标签
        min_cooccur: 最小共现次数阈值
    """
    print("\n" + "="*60)
    print("步骤 2.5: 生成标签共现网络图")
    print("="*60)

    # 获取top标签
    all_tags = []
    for tags in df_tagged['标签列表']:
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)
    top_tags = set([tag for tag, _ in tag_counts.most_common(top_n)])

    # 计算标签共现
    cooccurrence = Counter()
    for tags in df_tagged['标签列表']:
        # 只考虑top标签
        tags_filtered = [t for t in tags if t in top_tags]
        # 统计两两共现
        for i, tag1 in enumerate(tags_filtered):
            for tag2 in tags_filtered[i+1:]:
                pair = tuple(sorted([tag1, tag2]))
                cooccurrence[pair] += 1

    # 创建网络图
    G = nx.Graph()

    # 添加节点
    for tag in top_tags:
        G.add_node(tag, weight=tag_counts[tag])

    # 添加边（共现次数 >= min_cooccur）
    for (tag1, tag2), count in cooccurrence.items():
        if count >= min_cooccur:
            G.add_edge(tag1, tag2, weight=count)

    if len(G.edges()) == 0:
        print(f"⚠️  没有足够的标签共现关系（阈值: {min_cooccur}）")
        return

    # 绘制网络图
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 节点大小根据频率
    node_sizes = [G.nodes[node]['weight'] * 5 for node in G.nodes()]

    # 边宽度根据共现次数
    edge_widths = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]

    # 绘制
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                           alpha=0.7, ax=ax)

    # 标签
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontproperties=font_prop, fontsize=10,
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                     facecolor='white', alpha=0.7))

    ax.set_title('标签共现网络 (边宽度=共现次数)', fontproperties=font_prop,
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_file = config.VIZ_DIR / "tag_cooccurrence.png"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存: {output_file}")


def generate_statistics(df_tagged):
    """生成统计报告

    Args:
        df_tagged: 标签数据 DataFrame
    """
    print("\n" + "="*60)
    print("步骤 2.6: 生成统计报告")
    print("="*60)

    # 标签频率统计
    all_tags = []
    for tags in df_tagged['标签列表']:
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)

    stats_df = pd.DataFrame([
        {'标签': tag, '出现次数': count, '占比(%)': count/len(df_tagged)*100}
        for tag, count in tag_counts.most_common()
    ])

    stats_file = config.OUTPUT_DIR / "tag_statistics.csv"
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 保存标签统计: {stats_file}")

    # 档期-标签分析
    schedule_cols = ['是否春节档', '是否国庆档', '是否五一档', '是否暑期档', '是否普通周末']
    schedule_names = ['春节档', '国庆档', '五一档', '暑期档', '普通周末']

    schedule_analysis = []
    for schedule_col, schedule_name in zip(schedule_cols, schedule_names):
        if schedule_col not in df_tagged.columns:
            continue

        schedule_movies = df_tagged[df_tagged[schedule_col] == 1]
        if len(schedule_movies) == 0:
            continue

        schedule_tags = []
        for tags in schedule_movies['标签列表']:
            schedule_tags.extend(tags)

        top_5_tags = Counter(schedule_tags).most_common(5)
        schedule_analysis.append({
            '档期': schedule_name,
            '电影数量': len(schedule_movies),
            'Top5标签': ', '.join([f'{t}({c})' for t, c in top_5_tags])
        })

    if schedule_analysis:
        schedule_df = pd.DataFrame(schedule_analysis)
        schedule_file = config.OUTPUT_DIR / "schedule_tag_analysis.csv"
        schedule_df.to_csv(schedule_file, index=False, encoding='utf-8-sig')
        print(f"✓ 保存档期分析: {schedule_file}")

    # 打印关键洞察
    print("\n" + "="*60)
    print("关键洞察:")
    print("="*60)
    print(f"1. 最常见标签 Top 5:")
    for tag, count in tag_counts.most_common(5):
        print(f"   - {tag}: {count}次 ({count/len(df_tagged)*100:.1f}%)")

    if schedule_analysis:
        print(f"\n2. 各档期特征标签:")
        for item in schedule_analysis:
            print(f"   - {item['档期']} ({item['电影数量']}部): {item['Top5标签']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成LLM分析可视化和洞察")
    parser.add_argument('--top-n', type=int, default=20, help='显示前N个标签（默认20）')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM 分析可视化与洞察生成")
    print("="*60)

    # 配置中文字体
    font_prop = configure_chinese_font()

    # 1. 加载数据
    df_tagged = load_tagged_data()

    # 2. 生成可视化
    generate_tag_distribution(df_tagged, font_prop, top_n=args.top_n)
    generate_schedule_tag_heatmap(df_tagged, font_prop, top_n=15)
    generate_tag_performance(df_tagged, font_prop, top_n=args.top_n)
    generate_tag_cooccurrence(df_tagged, font_prop, top_n=15, min_cooccur=5)

    # 3. 生成统计报告
    generate_statistics(df_tagged)

    print("\n" + "="*60)
    print("✓ 完成！所有可视化和报告已生成")
    print("="*60)
    print(f"\n查看结果:")
    print(f"  - 可视化图表: {config.VIZ_DIR}")
    print(f"  - 统计报告: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
