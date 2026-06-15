#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清除 matplotlib 字体缓存

运行此脚本后再运行 EDA 脚本，可以解决中文字体显示问题
"""

import os
import shutil
from pathlib import Path

def clear_matplotlib_cache():
    """清除 matplotlib 字体缓存"""
    import matplotlib

    cache_dir = Path(matplotlib.get_cachedir())

    print("="*60)
    print("清除 matplotlib 字体缓存")
    print("="*60)
    print(f"\n缓存目录: {cache_dir}")

    if cache_dir.exists():
        # 查找并删除字体缓存文件
        font_cache_files = list(cache_dir.glob("*.cache")) + list(cache_dir.glob("fontlist-*.json"))

        if font_cache_files:
            print(f"\n找到 {len(font_cache_files)} 个缓存文件:")
            for f in font_cache_files:
                print(f"  - {f.name}")
                try:
                    f.unlink()
                    print(f"    ✓ 已删除")
                except Exception as e:
                    print(f"    ✗ 删除失败: {e}")
        else:
            print("\n未找到字体缓存文件")
    else:
        print("\n缓存目录不存在")

    print("\n" + "="*60)
    print("✓ 字体缓存清除完成")
    print("现在可以运行: python src/7_eda_visualization.py")
    print("="*60)

if __name__ == "__main__":
    clear_matplotlib_cache()
