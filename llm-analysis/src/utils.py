# -*- coding: utf-8 -*-
"""
工具函数模块
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_api_key():
    """从环境变量加载API密钥"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "请在 .env 文件中设置 DEEPSEEK_API_KEY\n"
            "如果还没有 API Key，请参考 README.md 中的申请指南"
        )
    return api_key


def configure_chinese_font():
    """配置中文字体"""
    font_path = "C:/Windows/Fonts/msyh.ttc"  # Microsoft YaHei
    if Path(font_path).exists():
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop
    else:
        # 尝试其他常见中文字体
        fallback_fonts = [
            "C:/Windows/Fonts/simhei.ttf",  # SimHei
            "C:/Windows/Fonts/simsun.ttc",  # SimSun
        ]
        for fallback in fallback_fonts:
            if Path(fallback).exists():
                font_prop = font_manager.FontProperties(fname=fallback)
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
                plt.rcParams['axes.unicode_minus'] = False
                return font_prop

        raise FileNotFoundError(
            f"未找到中文字体文件。尝试过的路径:\n"
            f"  - {font_path}\n"
            f"  - {', '.join(fallback_fonts)}"
        )


def estimate_cost(num_movies, avg_summary_length=200):
    """估算API调用成本

    Args:
        num_movies: 电影数量
        avg_summary_length: 平均剧情简介长度（字符数）

    Returns:
        dict: 包含成本估算的字典
    """
    # DeepSeek pricing (2026):
    # Input: ~¥1.00 per 1M tokens (~$0.14/1M tokens)
    # Output: ~¥2.00 per 1M tokens (~$0.28/1M tokens)

    # Token估算：
    # - 中文: 1个字符 ≈ 1.5 tokens
    # - 平均剧情简介: 200字符 ≈ 300 tokens
    # - 系统提示 + 用户提示开销: 每部电影约100 tokens
    # - 输出: 每部电影约50 tokens (JSON格式标签)

    tokens_per_movie_input = int(avg_summary_length * 1.5) + 100
    tokens_per_movie_output = 50

    total_input_tokens = num_movies * tokens_per_movie_input
    total_output_tokens = num_movies * tokens_per_movie_output

    # DeepSeek pricing
    input_cost_cny = (total_input_tokens / 1_000_000) * 1.00
    output_cost_cny = (total_output_tokens / 1_000_000) * 2.00
    total_cost_cny = input_cost_cny + output_cost_cny

    # Convert to USD (approximate rate)
    usd_to_cny = 7.2
    total_cost_usd = total_cost_cny / usd_to_cny

    return {
        "num_movies": num_movies,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "input_cost_cny": input_cost_cny,
        "output_cost_cny": output_cost_cny,
        "total_cost_cny": total_cost_cny,
        "total_cost_usd": total_cost_usd
    }


def print_cost_estimate(num_movies):
    """打印成本估算"""
    estimate = estimate_cost(num_movies)
    print(f"\n{'='*60}")
    print("💰 成本估算")
    print(f"{'='*60}")
    print(f"待处理电影数量: {estimate['num_movies']:,}")
    print(f"预计输入 Tokens: {estimate['estimated_input_tokens']:,}")
    print(f"预计输出 Tokens: {estimate['estimated_output_tokens']:,}")
    print(f"\n预计成本:")
    print(f"  输入: ¥{estimate['input_cost_cny']:.2f} CNY")
    print(f"  输出: ¥{estimate['output_cost_cny']:.2f} CNY")
    print(f"  ────────────────")
    print(f"  总计: ¥{estimate['total_cost_cny']:.2f} CNY (约 ${estimate['total_cost_usd']:.2f} USD)")
    print(f"{'='*60}\n")


def validate_tags(tags, movie_title):
    """验证标签是否在预定义列表中

    Args:
        tags: 标签列表
        movie_title: 电影标题（用于日志）

    Returns:
        list: 验证后的标签列表
    """
    if not isinstance(tags, list):
        print(f"⚠️  [{movie_title}] 标签格式错误，应为列表: {tags}")
        return []

    valid_tags = []
    for tag in tags:
        if tag in config.ALL_TAGS:
            valid_tags.append(tag)
        else:
            print(f"⚠️  [{movie_title}] 未知标签: {tag}")

    return valid_tags


def truncate_summary(summary, max_length=500):
    """截断过长的剧情简介

    Args:
        summary: 原始剧情简介
        max_length: 最大长度（字符数）

    Returns:
        str: 截断后的简介
    """
    if len(summary) <= max_length:
        return summary
    return summary[:max_length] + "..."
