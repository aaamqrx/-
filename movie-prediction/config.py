# -*- coding: utf-8 -*-
"""
电影预测模型配置文件
"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据路径配置
DATA_SOURCE = BASE_DIR.parent / "doban-data" / "data" / "final" / "china_all.xlsx"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
EVALUATION_DIR = BASE_DIR / "models" / "evaluation"

# 创建必要的目录
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 特征列配置
SCHEDULE_FEATURES = [
    '是否春节档',
    '是否国庆档',
    '是否五一档',
    '是否暑期档',
    '是否普通周末'
]

NUMERIC_FEATURES = [
    '上映前7天百度搜索指数均值',
    '百度指数标准差',
    '百度指数趋势斜率',
    '百度指数最大值',
    '百度指数最小值',
    '片长'
]

ALL_FEATURES = SCHEDULE_FEATURES + NUMERIC_FEATURES

# 细分类型特征会在 data_preparation 阶段通过 one-hot 编码动态添加
# 保留此列表用于后续引用
GENRE_FEATURE_PREFIX = '类型_'

# 目标变量
TARGET_REVIEWS = '评价人数'
TARGET_RATING = '评分'

# 数据分割配置
TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2  # 从训练集中分出验证集

# 模型超参数配置
LINEAR_REGRESSION_PARAMS = {
    'fit_intercept': True,
    'copy_X': True
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# 交叉验证配置
CV_FOLDS = 5

# 可视化配置
FIGURE_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
PLOT_FONT_SIZE = 10
CHINESE_FONT = 'Microsoft YaHei'  # 中文字体

# 日志配置
VERBOSE = True
