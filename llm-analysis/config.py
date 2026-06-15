# -*- coding: utf-8 -*-
"""
LLM 电影剧情分析配置文件
"""
from pathlib import Path

# 项目路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
VIZ_DIR = BASE_DIR / "visualizations"

# 数据源（从 doban-data 项目读取）
DATA_SOURCE = BASE_DIR.parent / "doban-data" / "data" / "final" / "china_all.xlsx"

# DeepSeek API配置
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"  # V4 Flash（非思考模式适合批量打标签）；旧名 deepseek-chat 将于 2026/07/24 弃用
DEEPSEEK_API_KEY = ""  # 从环境变量读取

# 处理配置
BATCH_SIZE = 10  # 每批处理的电影数量
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 60
MIN_SUMMARY_LENGTH = 20  # 最小剧情简介长度
MAX_CONCURRENT_BATCHES = 8  # 并发 API 请求数（DeepSeek 免费版超过此数可能限流）

# 预定义语义标签（按类型分组）
PREDEFINED_TAGS = {
    "情感": ["催泪", "温馨", "感动", "虐心", "治愈"],
    "风格": ["喜剧", "黑色幽默", "荒诞", "文艺", "商业"],
    "类型": ["动作", "悬疑", "惊悚", "科幻", "奇幻", "爱情", "犯罪"],
    "主题": ["主旋律", "青春", "励志", "家庭", "历史", "战争", "都市"],
    "受众": ["合家欢", "成人向", "青少年", "儿童"],
    "节奏": ["烧脑", "紧张刺激", "轻松搞笑", "沉重压抑"]
}

# 所有标签列表（用于LLM提示）
ALL_TAGS = [tag for category in PREDEFINED_TAGS.values() for tag in category]

# 可视化配置
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 300
CHINESE_FONT = "Microsoft YaHei"

# 确保目录存在
for d in [DATA_DIR, OUTPUT_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)
