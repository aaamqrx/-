# -*- coding: utf-8 -*-
"""
步骤3: 模型训练

功能:
1. 加载特征工程后的数据
2. 训练线性回归模型
3. 训练随机森林模型
4. 交叉验证
5. 保存训练好的模型
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import print_metrics, calculate_metrics, save_model

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_engineered_data():
    """加载特征工程后的数据"""
    print("\n" + "="*60)
    print("步骤 3.1: 加载特征工程数据")
    print("="*60)

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_scaled.csv")
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test_scaled.csv")

    y_reviews_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_reviews_train_log.csv").squeeze()
    y_reviews_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_reviews_test_log.csv").squeeze()

    y_rating_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_rating_train_final.csv").squeeze()
    y_rating_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_rating_test_final.csv").squeeze()

    print(f"训练集特征: {X_train.shape}")
    print(f"测试集特征: {X_test.shape}")

    return X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test


def train_linear_regression(X_train, y_train, task_name="评价人数"):
    """训练线性回归模型"""
    print(f"\n训练线性回归模型 - {task_name}")
    print("-" * 60)

    # 创建模型
    model = LinearRegression(**config.LINEAR_REGRESSION_PARAMS)

    # 训练
    model.fit(X_train, y_train)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring='r2')

    print(f"✓ 模型训练完成")
    print(f"  交叉验证 R² 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 打印特征系数
    print(f"\n特征系数:")
    feature_names = X_train.columns.tolist()
    for feature, coef in zip(feature_names, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  截距: {model.intercept_:.4f}")

    return model


def train_random_forest(X_train, y_train, task_name="评价人数"):
    """训练随机森林模型"""
    print(f"\n训练随机森林模型 - {task_name}")
    print("-" * 60)

    # 创建模型
    model = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)

    # 训练
    model.fit(X_train, y_train)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring='r2')

    print(f"✓ 模型训练完成")
    print(f"  交叉验证 R² 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 打印特征重要性
    print(f"\n特征重要性:")
    feature_names = X_train.columns.tolist()
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, importance in feature_importance:
        print(f"  {feature}: {importance:.4f}")

    return model


def evaluate_on_test(model, X_test, y_test, task_name="", inverse_log=False):
    """在测试集上评估模型"""
    y_pred = model.predict(X_test)

    # 如果需要反对数变换
    if inverse_log:
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        metrics = calculate_metrics(y_test_original, y_pred_original, prefix=task_name)
    else:
        metrics = calculate_metrics(y_test, y_pred, prefix=task_name)

    return metrics


def main():
    """主函数"""
    print("\n" + "="*60)
    print("电影预测模型 - 模型训练")
    print("="*60)

    # 1. 加载数据
    X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test = load_engineered_data()

    # ==================== 任务1: 预测评价人数 ====================
    print("\n" + "="*60)
    print("任务 1: 预测评价人数（热度/票房代理）")
    print("="*60)

    # 训练线性回归
    lr_reviews = train_linear_regression(X_train, y_reviews_train, "评价人数")
    lr_reviews_metrics = evaluate_on_test(lr_reviews, X_test, y_reviews_test, "LR_Reviews_", inverse_log=True)
    print_metrics(lr_reviews_metrics, "线性回归 - 评价人数测试集表现")

    # 训练随机森林
    rf_reviews = train_random_forest(X_train, y_reviews_train, "评价人数")
    rf_reviews_metrics = evaluate_on_test(rf_reviews, X_test, y_reviews_test, "RF_Reviews_", inverse_log=True)
    print_metrics(rf_reviews_metrics, "随机森林 - 评价人数测试集表现")

    # 保存模型
    save_model(lr_reviews, config.MODELS_DIR / "lr_reviews.pkl")
    save_model(rf_reviews, config.MODELS_DIR / "rf_reviews.pkl")

    # ==================== 任务2: 预测评分 ====================
    print("\n" + "="*60)
    print("任务 2: 预测豆瓣评分（质量）")
    print("="*60)

    # 训练线性回归
    lr_rating = train_linear_regression(X_train, y_rating_train, "评分")
    lr_rating_metrics = evaluate_on_test(lr_rating, X_test, y_rating_test, "LR_Rating_", inverse_log=False)
    print_metrics(lr_rating_metrics, "线性回归 - 评分测试集表现")

    # 训练随机森林
    rf_rating = train_random_forest(X_train, y_rating_train, "评分")
    rf_rating_metrics = evaluate_on_test(rf_rating, X_test, y_rating_test, "RF_Rating_", inverse_log=False)
    print_metrics(rf_rating_metrics, "随机森林 - 评分测试集表现")

    # 保存模型
    save_model(lr_rating, config.MODELS_DIR / "lr_rating.pkl")
    save_model(rf_rating, config.MODELS_DIR / "rf_rating.pkl")

    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("✓ 模型训练完成！")
    print("="*60)
    print(f"\n已保存的模型:")
    print(f"  1. lr_reviews.pkl - 线性回归预测评价人数")
    print(f"  2. rf_reviews.pkl - 随机森林预测评价人数")
    print(f"  3. lr_rating.pkl - 线性回归预测评分")
    print(f"  4. rf_rating.pkl - 随机森林预测评分")


if __name__ == "__main__":
    main()
