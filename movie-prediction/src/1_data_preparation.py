# -*- coding: utf-8 -*-
"""
步骤1: 数据准备

功能:
1. 加载原始数据
2. 筛选有效百度指数的记录
3. 处理缺失值
4. 数据分割
5. 保存处理后的数据
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import parse_runtime, print_data_summary

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data():
    """加载原始数据"""
    print("\n" + "="*60)
    print("步骤 1.1: 加载原始数据")
    print("="*60)

    print(f"数据源: {config.DATA_SOURCE}")

    if not config.DATA_SOURCE.exists():
        raise FileNotFoundError(f"数据文件不存在: {config.DATA_SOURCE}")

    df = pd.read_excel(config.DATA_SOURCE)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()[:10]}...")

    return df


def filter_valid_records(df):
    """筛选有有效百度指数的记录"""
    print("\n" + "="*60)
    print("步骤 1.2: 筛选有效记录")
    print("="*60)

    initial_count = len(df)

    # 筛选有有效百度指数的记录
    baidu_col = '上映前7天百度搜索指数均值'
    df_filtered = df[df[baidu_col].notna() & (df[baidu_col] > 0)].copy()

    print(f"有效百度指数记录: {len(df_filtered)}/{initial_count} ({len(df_filtered)/initial_count*100:.1f}%)")

    # 筛选有评分和评价人数的记录
    df_filtered = df_filtered[
        df_filtered['评分'].notna() &
        df_filtered['评价人数'].notna() &
        (df_filtered['评分'] > 0) &
        (df_filtered['评价人数'] > 0)
    ]

    print(f"同时有评分和评价人数: {len(df_filtered)}/{initial_count}")

    return df_filtered


def process_features(df):
    """处理特征"""
    print("\n" + "="*60)
    print("步骤 1.3: 处理特征")
    print("="*60)

    df_processed = df.copy()

    # 处理片长
    print("处理片长字段...")
    df_processed['片长_数值'] = df_processed['片长'].apply(parse_runtime)

    # 检查片长缺失情况
    missing_runtime = df_processed['片长_数值'].isna().sum()
    print(f"  片长缺失: {missing_runtime}/{len(df_processed)} ({missing_runtime/len(df_processed)*100:.1f}%)")

    # 用中位数填充缺失的片长
    # 注意：必须用赋值方式，不能用 df[col].fillna(..., inplace=True)。
    # 在 pandas 2.x/3.0 的 Copy-on-Write 下，链式 inplace 会作用在临时副本上被丢弃，导致填充无效。
    if missing_runtime > 0:
        median_runtime = df_processed['片长_数值'].median()
        df_processed['片长_数值'] = df_processed['片长_数值'].fillna(median_runtime)
        print(f"  用中位数填充片长缺失值: {median_runtime:.0f}分钟")

    # 确保档期特征为数值类型
    for col in config.SCHEDULE_FEATURES:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0).astype(int)

    # 兜底：确保所有数值特征都没有NaN
    numeric_cols = ['上映前7天百度搜索指数均值', '片长_数值']
    for col in numeric_cols:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                print(f"  兜底填充 {col}: {nan_count} 个缺失值，使用中位数 {median_val:.2f}")

    # 最终检查
    print("\n最终NaN检查:")
    nan_check = df_processed[config.SCHEDULE_FEATURES + ['上映前7天百度搜索指数均值', '片长_数值']].isna().sum()
    if nan_check.sum() > 0:
        print("⚠ 警告：仍有缺失值！")
        print(nan_check[nan_check > 0])
    else:
        print("  ✓ 所有特征均无缺失值")

    return df_processed


def select_columns(df):
    """选择需要的列"""
    print("\n" + "="*60)
    print("步骤 1.4: 选择特征和目标变量")
    print("="*60)

    # 基础特征列表
    base_feature_cols = config.SCHEDULE_FEATURES + [
        '上映前7天百度搜索指数均值',
        '百度指数标准差',
        '百度指数趋势斜率',
        '百度指数最大值',
        '百度指数最小值',
        '片长_数值'
    ]
    target_cols = [config.TARGET_REVIEWS, config.TARGET_RATING]

    # 检查原始数据中是否存在新增的百度衍生特征列
    available_base = [c for c in base_feature_cols if c in df.columns]
    missing_base = [c for c in base_feature_cols if c not in df.columns]
    if missing_base:
        print(f"⚠ 缺失特征列（将跳过）：{missing_base}")

    # 选择所需列（保留"类型"用于 one-hot 编码）
    type_col = '类型' if '类型' in df.columns else None
    selected_cols = available_base + target_cols
    if type_col:
        selected_cols.append(type_col)
    df_selected = df[selected_cols].copy()

    # 重命名片长_数值为片长
    if '片长_数值' in df_selected.columns:
        df_selected.rename(columns={'片长_数值': '片长'}, inplace=True)

    print(f"基础特征列: {available_base}")
    print(f"目标列: {target_cols}")
    print(f"最终数据形状: {df_selected.shape}")

    return df_selected


def encode_genres(df, feature_cols):
    """对'类型'列做 one-hot 编码，返回更新后的特征列名列表"""
    if '类型' not in df.columns:
        print("  (无类型列，跳过 one-hot 编码)")
        return df, feature_cols

    print("对细分类型做 one-hot 编码...")
    # 拆分逗号分隔的类型字符串，展开为多行
    genre_series = df['类型'].fillna('').astype(str)
    all_genres = set()
    for text in genre_series:
        for g in text.split(','):
            g = g.strip()
            if g:
                all_genres.add(g)

    if not all_genres:
        print("  (类型列为空，跳过)")
        df = df.drop(columns=['类型'])
        return df, feature_cols

    # 为每个类型创建 one-hot 列
    new_cols = []
    for genre in sorted(all_genres):
        col_name = f"{config.GENRE_FEATURE_PREFIX}{genre}"
        df[col_name] = genre_series.apply(lambda s: 1 if genre in [x.strip() for x in s.split(',')] else 0).astype(int)
        new_cols.append(col_name)

    # 删除原始"类型"列
    df = df.drop(columns=['类型'])

    updated_features = feature_cols + new_cols
    print(f"  新增 {len(new_cols)} 个类型特征列: {new_cols[:10]}..." if len(new_cols) > 10 else f"  新增 {len(new_cols)} 个类型特征列: {new_cols}")

    return df, updated_features


def split_data(df, feature_cols):
    """分割训练集和测试集"""
    print("\n" + "="*60)
    print("步骤 1.5: 分割数据集")
    print("="*60)

    # 准备特征和目标
    # 确保所有 feature_cols 都存在
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].copy()
    y_reviews = df[config.TARGET_REVIEWS].copy()
    y_rating = df[config.TARGET_RATING].copy()

    # 分割数据
    X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test = train_test_split(
        X, y_reviews, y_rating,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"训练集大小: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"测试集大小: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    return X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test, available_features


def save_processed_data(X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test):
    """保存处理后的数据"""
    print("\n" + "="*60)
    print("步骤 1.6: 保存处理后的数据")
    print("="*60)

    # 保存数据
    X_train.to_csv(config.PROCESSED_DATA_DIR / "X_train.csv", index=False, encoding='utf-8-sig')
    X_test.to_csv(config.PROCESSED_DATA_DIR / "X_test.csv", index=False, encoding='utf-8-sig')

    y_reviews_train.to_csv(config.PROCESSED_DATA_DIR / "y_reviews_train.csv", index=False, header=True)
    y_reviews_test.to_csv(config.PROCESSED_DATA_DIR / "y_reviews_test.csv", index=False, header=True)

    y_rating_train.to_csv(config.PROCESSED_DATA_DIR / "y_rating_train.csv", index=False, header=True)
    y_rating_test.to_csv(config.PROCESSED_DATA_DIR / "y_rating_test.csv", index=False, header=True)

    print(f"✓ 数据已保存到: {config.PROCESSED_DATA_DIR}")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_reviews_train.csv: {y_reviews_train.shape}")
    print(f"  - y_reviews_test.csv: {y_reviews_test.shape}")
    print(f"  - y_rating_train.csv: {y_rating_train.shape}")
    print(f"  - y_rating_test.csv: {y_rating_test.shape}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("电影预测模型 - 数据准备")
    print("="*60)

    # 1. 加载原始数据
    df_raw = load_raw_data()

    # 2. 筛选有效记录
    df_filtered = filter_valid_records(df_raw)

    # 3. 处理特征
    df_processed = process_features(df_filtered)

    # 4. 选择列
    df_final = select_columns(df_processed)

    # 4.5 类型 one-hot 编码
    current_features = [c for c in config.ALL_FEATURES if c in df_final.columns]
    df_final, current_features = encode_genres(df_final, current_features)

    # 5. 打印数据摘要
    print_data_summary(df_final, "处理后的数据摘要")

    # 6. 分割数据
    X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test, final_features = split_data(df_final, current_features)

    # 7. 保存数据
    save_processed_data(X_train, X_test, y_reviews_train, y_reviews_test, y_rating_train, y_rating_test)

    print("\n" + "="*60)
    print("✓ 数据准备完成！")
    print("="*60)


if __name__ == "__main__":
    main()
