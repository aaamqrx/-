# -*- coding: utf-8 -*-
"""
步骤2: 特征工程

功能:
1. 加载处理后的数据
2. 特征标准化/归一化
3. 目标变量转换（对数变换）
4. 保存转换器和处理后的特征
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import print_data_summary

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def load_processed_data():
    """加载处理后的数据"""
    print("\n" + "="*60)
    print("步骤 2.1: 加载处理后的数据")
    print("="*60)

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")

    y_reviews_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_reviews_train.csv").squeeze()
    y_reviews_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_reviews_test.csv").squeeze()

    y_rating_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_rating_train.csv").squeeze()
    y_rating_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_rating_test.csv").squeeze()

    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")

    return X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test


def scale_features(X_train, X_test):
    """特征标准化"""
    print("\n" + "="*60)
    print("步骤 2.2: 特征标准化")
    print("="*60)

    # 创建标准化器
    scaler = StandardScaler()

    # 只对数值特征进行标准化（档期和类型特征保持0/1）
    # 过滤出实际存在的数值特征列
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in X_train.columns]

    if not numeric_cols:
        print("⚠ 没有可标准化的数值特征列，跳过标准化")
        return X_train.copy(), X_test.copy(), scaler

    # 拟合训练集并转换
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # 转换测试集
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"标准化的特征: {numeric_cols}")
    print(f"训练集特征均值: {X_train_scaled[numeric_cols].mean().values}")
    print(f"训练集特征标准差: {X_train_scaled[numeric_cols].std().values}")

    # 保存标准化器
    scaler_path = config.MODELS_DIR / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ 标准化器已保存: {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def transform_target_reviews(y_reviews_train, y_reviews_test):
    """对评价人数进行对数转换"""
    print("\n" + "="*60)
    print("步骤 2.3: 目标变量转换 - 评价人数")
    print("="*60)

    print(f"原始评价人数 - 训练集统计:")
    print(f"  均值: {y_reviews_train.mean():.2f}")
    print(f"  中位数: {y_reviews_train.median():.2f}")
    print(f"  标准差: {y_reviews_train.std():.2f}")
    print(f"  最小值: {y_reviews_train.min():.2f}")
    print(f"  最大值: {y_reviews_train.max():.2f}")

    # 对数转换 log(x + 1)
    y_reviews_train_log = np.log1p(y_reviews_train)
    y_reviews_test_log = np.log1p(y_reviews_test)

    print(f"\n对数转换后 - 训练集统计:")
    print(f"  均值: {y_reviews_train_log.mean():.2f}")
    print(f"  中位数: {y_reviews_train_log.median():.2f}")
    print(f"  标准差: {y_reviews_train_log.std():.2f}")
    print(f"  最小值: {y_reviews_train_log.min():.2f}")
    print(f"  最大值: {y_reviews_train_log.max():.2f}")

    return y_reviews_train_log, y_reviews_test_log


def save_engineered_data(X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test):
    """保存特征工程后的数据"""
    print("\n" + "="*60)
    print("步骤 2.4: 保存特征工程后的数据")
    print("="*60)

    # 保存标准化后的特征
    X_train.to_csv(config.PROCESSED_DATA_DIR / "X_train_scaled.csv", index=False, encoding='utf-8-sig')
    X_test.to_csv(config.PROCESSED_DATA_DIR / "X_test_scaled.csv", index=False, encoding='utf-8-sig')

    # 保存转换后的目标变量
    pd.Series(y_reviews_train, name=config.TARGET_REVIEWS).to_csv(
        config.PROCESSED_DATA_DIR / "y_reviews_train_log.csv", index=False, header=True
    )
    pd.Series(y_reviews_test, name=config.TARGET_REVIEWS).to_csv(
        config.PROCESSED_DATA_DIR / "y_reviews_test_log.csv", index=False, header=True
    )

    # 评分不需要转换，直接复制
    pd.Series(y_rating_train, name=config.TARGET_RATING).to_csv(
        config.PROCESSED_DATA_DIR / "y_rating_train_final.csv", index=False, header=True
    )
    pd.Series(y_rating_test, name=config.TARGET_RATING).to_csv(
        config.PROCESSED_DATA_DIR / "y_rating_test_final.csv", index=False, header=True
    )

    print(f"✓ 特征工程数据已保存到: {config.PROCESSED_DATA_DIR}")
    print(f"  - X_train_scaled.csv")
    print(f"  - X_test_scaled.csv")
    print(f"  - y_reviews_train_log.csv (对数转换)")
    print(f"  - y_reviews_test_log.csv (对数转换)")
    print(f"  - y_rating_train_final.csv (原始)")
    print(f"  - y_rating_test_final.csv (原始)")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("电影预测模型 - 特征工程")
    print("="*60)

    # 1. 加载数据
    X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test = load_processed_data()

    # 2. 特征标准化
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 3. 目标变量转换
    y_reviews_train_log, y_reviews_test_log = transform_target_reviews(y_reviews_train, y_reviews_test)

    # 评分不需要转换
    print("\n" + "="*60)
    print("步骤 2.3b: 评分目标变量（不转换）")
    print("="*60)
    print(f"评分 - 训练集统计:")
    print(f"  均值: {y_rating_train.mean():.2f}")
    print(f"  中位数: {y_rating_train.median():.2f}")
    print(f"  标准差: {y_rating_train.std():.2f}")
    print(f"  最小值: {y_rating_train.min():.2f}")
    print(f"  最大值: {y_rating_train.max():.2f}")

    # 4. 保存数据
    save_engineered_data(
        X_train_scaled, X_test_scaled,
        y_reviews_train_log, y_reviews_test_log,
        y_rating_train, y_rating_test
    )

    print("\n" + "="*60)
    print("✓ 特征工程完成！")
    print("="*60)


if __name__ == "__main__":
    main()
