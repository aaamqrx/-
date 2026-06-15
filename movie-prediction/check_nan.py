# -*- coding: utf-8 -*-
"""
临时诊断脚本 - 检查数据中的NaN
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import pandas as pd

print("检查处理后的数据...")

# 检查训练数据
X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_scaled.csv")

print(f"\nX_train形状: {X_train.shape}")
print(f"\nX_train列名: {X_train.columns.tolist()}")
print(f"\nNaN统计:")
print(X_train.isna().sum())

if X_train.isna().sum().sum() > 0:
    print("\n⚠ 发现NaN！显示包含NaN的行:")
    nan_rows = X_train[X_train.isna().any(axis=1)]
    print(nan_rows.head(10))
else:
    print("\n✓ 数据干净，无NaN")

print("\n数据样本:")
print(X_train.head())
