# -*- coding: utf-8 -*-
"""
工具函数模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path


def configure_chinese_font(font_name='Microsoft YaHei'):
    """配置matplotlib中文字体"""
    # Windows 系统字体目录
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("C:/Windows/Fonts")
    ]

    # 按优先级查找字体文件
    font_files = [
        ('msyh.ttc', 'Microsoft YaHei'),
        ('msyhbd.ttc', 'Microsoft YaHei'),
        ('simhei.ttf', 'SimHei'),
        ('simsun.ttc', 'SimSun'),
    ]

    font_path = None
    for font_file, _ in font_files:
        for font_dir in font_dirs:
            test_path = font_dir / font_file
            if test_path.exists():
                font_path = str(test_path)
                break
        if font_path:
            break

    if font_path:
        from matplotlib.font_manager import FontProperties
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop
    else:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return None


def parse_runtime(runtime_str):
    """
    解析片长字符串，转换为分钟数

    输入示例: "120分钟", "2小时30分钟", "90分钟 / 85分钟(中国大陆)"
    输出: 数值（分钟）
    """
    if pd.isna(runtime_str):
        return np.nan

    runtime_str = str(runtime_str).strip()

    # 提取第一个时长（处理多个时长的情况）
    if '/' in runtime_str:
        runtime_str = runtime_str.split('/')[0].strip()

    # 移除括号内容
    if '(' in runtime_str:
        runtime_str = runtime_str.split('(')[0].strip()

    total_minutes = 0

    # 处理"X小时Y分钟"格式
    if '小时' in runtime_str:
        parts = runtime_str.split('小时')
        try:
            hours = int(''.join(filter(str.isdigit, parts[0])))
            total_minutes += hours * 60
            if len(parts) > 1 and '分钟' in parts[1]:
                minutes = int(''.join(filter(str.isdigit, parts[1].split('分钟')[0])))
                total_minutes += minutes
        except:
            return np.nan
    # 处理"X分钟"格式
    elif '分钟' in runtime_str:
        try:
            minutes = int(''.join(filter(str.isdigit, runtime_str.split('分钟')[0])))
            total_minutes = minutes
        except:
            return np.nan
    # 尝试直接提取数字
    else:
        try:
            total_minutes = int(''.join(filter(str.isdigit, runtime_str)))
        except:
            return np.nan

    # 合理性检查：电影片长通常在30-300分钟之间
    if 30 <= total_minutes <= 300:
        return total_minutes
    else:
        return np.nan


def print_data_summary(df, title="数据摘要"):
    """打印数据集摘要信息"""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"样本数: {len(df)}")
    print(f"特征数: {df.shape[1]}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  无缺失值")
    print("="*60)


def calculate_metrics(y_true, y_pred, prefix=""):
    """
    计算回归评估指标

    返回: dict with RMSE, MAE, R2, MAPE
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (避免除零)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}MAPE': mape
    }

    return metrics


def print_metrics(metrics, title="模型评估指标"):
    """打印评估指标"""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'MAPE' in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)


def save_model(model, filepath):
    """保存模型到文件"""
    import joblib
    joblib.dump(model, filepath)
    print(f"✓ 模型已保存: {filepath}")


def load_model(filepath):
    """从文件加载模型"""
    import joblib
    model = joblib.load(filepath)
    print(f"✓ 模型已加载: {filepath}")
    return model
