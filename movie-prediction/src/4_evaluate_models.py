# -*- coding: utf-8 -*-
"""
步骤4: 模型评估

功能:
1. 加载训练好的模型
2. 在测试集上评估
3. 生成可视化图表
4. 保存评估报告
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import load_model, calculate_metrics, print_metrics, configure_chinese_font

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_test_data():
    """加载测试数据"""
    print("\n" + "="*60)
    print("步骤 4.1: 加载测试数据")
    print("="*60)

    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_reviews_test_log = pd.read_csv(config.PROCESSED_DATA_DIR / "y_reviews_test_log.csv").squeeze()
    y_rating_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_rating_test_final.csv").squeeze()

    # 反对数变换评价人数
    y_reviews_test = np.expm1(y_reviews_test_log)

    print(f"测试集特征: {X_test.shape}")
    print(f"测试集样本数: {len(X_test)}")

    return X_test, y_reviews_test, y_reviews_test_log, y_rating_test


def load_all_models():
    """加载所有训练好的模型"""
    print("\n" + "="*60)
    print("步骤 4.2: 加载训练好的模型")
    print("="*60)

    models = {
        'lr_reviews': load_model(config.MODELS_DIR / "lr_reviews.pkl"),
        'rf_reviews': load_model(config.MODELS_DIR / "rf_reviews.pkl"),
        'lr_rating': load_model(config.MODELS_DIR / "lr_rating.pkl"),
        'rf_rating': load_model(config.MODELS_DIR / "rf_rating.pkl")
    }

    return models


def evaluate_models(models, X_test, y_reviews_test, y_reviews_test_log, y_rating_test):
    """评估所有模型"""
    print("\n" + "="*60)
    print("步骤 4.3: 评估模型性能")
    print("="*60)

    all_metrics = {}

    # 评价人数预测
    print("\n【任务1: 预测评价人数】")
    print("-" * 60)

    # 线性回归
    lr_reviews_pred_log = models['lr_reviews'].predict(X_test)
    lr_reviews_pred = np.expm1(lr_reviews_pred_log)
    lr_reviews_metrics = calculate_metrics(y_reviews_test, lr_reviews_pred, "LR_Reviews_")
    print_metrics(lr_reviews_metrics, "线性回归 - 评价人数")
    all_metrics.update(lr_reviews_metrics)

    # 随机森林
    rf_reviews_pred_log = models['rf_reviews'].predict(X_test)
    rf_reviews_pred = np.expm1(rf_reviews_pred_log)
    rf_reviews_metrics = calculate_metrics(y_reviews_test, rf_reviews_pred, "RF_Reviews_")
    print_metrics(rf_reviews_metrics, "随机森林 - 评价人数")
    all_metrics.update(rf_reviews_metrics)

    # 评分预测
    print("\n【任务2: 预测评分】")
    print("-" * 60)

    # 线性回归
    lr_rating_pred = models['lr_rating'].predict(X_test)
    lr_rating_metrics = calculate_metrics(y_rating_test, lr_rating_pred, "LR_Rating_")
    print_metrics(lr_rating_metrics, "线性回归 - 评分")
    all_metrics.update(lr_rating_metrics)

    # 随机森林
    rf_rating_pred = models['rf_rating'].predict(X_test)
    rf_rating_metrics = calculate_metrics(y_rating_test, rf_rating_pred, "RF_Rating_")
    print_metrics(rf_rating_metrics, "随机森林 - 评分")
    all_metrics.update(rf_rating_metrics)

    # 准备预测结果
    predictions = {
        'lr_reviews': lr_reviews_pred,
        'rf_reviews': rf_reviews_pred,
        'lr_rating': lr_rating_pred,
        'rf_rating': rf_rating_pred
    }

    return all_metrics, predictions


def plot_predictions_vs_actual(y_true, y_pred, title, output_file, font_prop=None):
    """绘制预测值vs实际值散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='none')

    # 理想预测线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')

    ax.set_xlabel('实际值', fontsize=12, fontproperties=font_prop)
    ax.set_ylabel('预测值', fontsize=12, fontproperties=font_prop)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(prop=font_prop)

    plt.tight_layout()
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存: {output_file}")


def plot_feature_importance(model, output_file, font_prop=None):
    """绘制特征重要性图"""
    if not hasattr(model, 'feature_importances_'):
        print("  该模型不支持特征重要性分析")
        return

    # 获取特征重要性，优先使用模型记录的特征名
    importance = model.feature_importances_
    if hasattr(model, 'feature_names_in_') and len(model.feature_names_in_) == len(importance):
        features = list(model.feature_names_in_)
    else:
        features = [f"特征{i}" for i in range(len(importance))]

    # 排序
    indices = np.argsort(importance)[::-1]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(indices)), importance[indices], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])

    # 设置y轴标签字体
    for label in ax.get_yticklabels():
        if font_prop:
            label.set_fontproperties(font_prop)

    ax.set_xlabel('重要性', fontsize=12, fontproperties=font_prop)
    ax.set_title('随机森林特征重要性', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存: {output_file}")


def generate_visualizations(models, y_reviews_test, y_rating_test, predictions, font_prop):
    """生成所有可视化图表"""
    print("\n" + "="*60)
    print("步骤 4.4: 生成可视化图表")
    print("="*60)

    # 1. 评价人数预测 - 线性回归
    plot_predictions_vs_actual(
        y_reviews_test, predictions['lr_reviews'],
        '线性回归：评价人数预测 vs 实际',
        config.EVALUATION_DIR / "pred_vs_actual_lr_reviews.png",
        font_prop
    )

    # 2. 评价人数预测 - 随机森林
    plot_predictions_vs_actual(
        y_reviews_test, predictions['rf_reviews'],
        '随机森林：评价人数预测 vs 实际',
        config.EVALUATION_DIR / "pred_vs_actual_rf_reviews.png",
        font_prop
    )

    # 3. 评分预测 - 线性回归
    plot_predictions_vs_actual(
        y_rating_test, predictions['lr_rating'],
        '线性回归：评分预测 vs 实际',
        config.EVALUATION_DIR / "pred_vs_actual_lr_rating.png",
        font_prop
    )

    # 4. 评分预测 - 随机森林
    plot_predictions_vs_actual(
        y_rating_test, predictions['rf_rating'],
        '随机森林：评分预测 vs 实际',
        config.EVALUATION_DIR / "pred_vs_actual_rf_rating.png",
        font_prop
    )

    # 5. 特征重要性 - 评价人数
    plot_feature_importance(
        models['rf_reviews'],
        config.EVALUATION_DIR / "feature_importance_reviews.png",
        font_prop
    )

    # 6. 特征重要性 - 评分
    plot_feature_importance(
        models['rf_rating'],
        config.EVALUATION_DIR / "feature_importance_rating.png",
        font_prop
    )


def save_metrics_summary(metrics):
    """保存评估指标汇总"""
    print("\n" + "="*60)
    print("步骤 4.5: 保存评估报告")
    print("="*60)

    # 转换为DataFrame
    df_metrics = pd.DataFrame([metrics])

    # 保存
    output_file = config.EVALUATION_DIR / "metrics_summary.csv"
    df_metrics.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✓ 评估指标已保存: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("电影预测模型 - 模型评估")
    print("="*60)

    # 配置中文字体
    font_prop = configure_chinese_font(config.CHINESE_FONT)

    # 设置绘图样式
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.dpi'] = config.FIGURE_DPI

    # 1. 加载测试数据
    X_test, y_reviews_test, y_reviews_test_log, y_rating_test = load_test_data()

    # 2. 加载模型
    models = load_all_models()

    # 3. 评估模型
    all_metrics, predictions = evaluate_models(models, X_test, y_reviews_test, y_reviews_test_log, y_rating_test)

    # 4. 生成可视化
    generate_visualizations(models, y_reviews_test, y_rating_test, predictions, font_prop)

    # 5. 保存指标
    save_metrics_summary(all_metrics)

    print("\n" + "="*60)
    print("✓ 模型评估完成！")
    print(f"所有结果已保存到: {config.EVALUATION_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
