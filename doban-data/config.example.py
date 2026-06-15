# -*- coding: utf-8 -*-
from pathlib import Path

# 豆瓣电影类型配置（type_id已全部补全）
MOVIE_TYPES = {
    "documentary": {"type_id": 1, "name": "纪录片"},
    "biography": {"type_id": 2, "name": "传记"},
    "crime": {"type_id": 3, "name": "犯罪"},
    "history": {"type_id": 4, "name": "历史"},
    "action": {"type_id": 5, "name": "动作"},
    "erotic": {"type_id": 6, "name": "情色"},
    "musical": {"type_id": 7, "name": "歌舞"},
    "children": {"type_id": 8, "name": "儿童"},
    "drama": {"type_id": 11, "name": "剧情"},
    "disaster": {"type_id": 12, "name": "灾难"},
    "romance": {"type_id": 13, "name": "爱情"},
    "music": {"type_id": 14, "name": "音乐"},
    "adventure": {"type_id": 15, "name": "冒险"},
    "fantasy": {"type_id": 16, "name": "奇幻"},
    "scifi": {"type_id": 17, "name": "科幻"},
    "sports": {"type_id": 18, "name": "运动"},
    "thriller": {"type_id": 19, "name": "惊悚"},
    "horror": {"type_id": 20, "name": "恐怖"},
    "war": {"type_id": 22, "name": "战争"},
    "short": {"type_id": 23, "name": "短片"},
    "comedy": {"type_id": 24, "name": "喜剧"},
    "western": {"type_id": 27, "name": "西部"},
    "family": {"type_id": 28, "name": "家庭"},
    "wuxia": {"type_id": 29, "name": "武侠"},
    "costume": {"type_id": 30, "name": "古装"},
    "noir": {"type_id": 31, "name": "黑色电影"},
    "suspense": {"type_id": 10, "name": "悬疑"},
}

# 中国地区关键词（包含大陆、香港、台湾）
CHINA_KEYWORDS = ["中国大陆", "大陆", "中国香港", "香港", "中国台湾", "台湾"]

# 目标数量（每个类型）
TARGET_CHINA_COUNT = 2000

# 分页配置
LIMIT_PER_PAGE = 20

# 常规请求延时
DELAY_MIN = 3
DELAY_MAX = 5

# HTML下载相关配置
REQUEST_TIMEOUT = 20
BATCH_SIZE = 50
BATCH_PAUSE_MIN = 120
BATCH_PAUSE_MAX = 300
BLOCK_RETRY_COUNT = 2
BLOCK_RETRY_PAUSES = (30, 90)
BLOCKED_THRESHOLD = 3
BLOCK_PAUSE_MIN = 300
BLOCK_PAUSE_MAX = 900
STOP_BLOCKED_THRESHOLD = 5

# 百度指数配置
# ⚠️ 重要：请将下面的 Cookie 替换为你自己的百度登录 Cookie
# 获取方法：浏览器登录百度后，打开开发者工具 → Application → Cookies → 复制整串 Cookie
BAIDU_INDEX_COOKIE = ""  # 在此填入你的百度 Cookie（多行字符串用 r'''...''' 包裹）
BAIDU_INDEX_AREA = 0
BAIDU_INDEX_MIN_DATE = "2011-01-01"
BAIDU_INDEX_PREFLIGHT_QUERY = "红海行动"
BAIDU_INDEX_PREFLIGHT_START_DATE = "2018-02-09"
BAIDU_INDEX_PREFLIGHT_END_DATE = "2018-02-15"
BAIDU_INDEX_LOOKBACK_DAYS = 7
BAIDU_INDEX_DELAY_MIN = 1
BAIDU_INDEX_DELAY_MAX = 2
BAIDU_INDEX_RETRY_COUNT = 2
BAIDU_INDEX_RETRY_PAUSE = 3
BAIDU_INDEX_OUTPUT_FIELD = "上映前7天百度搜索指数均值"
BAIDU_INDEX_ALL_ZERO_WARN_THRESHOLD = 0.9
BAIDU_INDEX_ERROR_WARN_THRESHOLD = 0.8
BAIDU_TITLE_SPLIT_PATTERN = r"[：:（(【\[\-—·]"

# 目录配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
HTML_DIR = BASE_DIR / "html_cache"
BAIDU_CACHE_DIR = BASE_DIR / "baidu_cache"
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
BAIDU_DIR = DATA_DIR / "baidu"
FINAL_DIR = DATA_DIR / "final"
SRC_DIR = BASE_DIR / "src"

# 确保目录存在
for d in [DATA_DIR, HTML_DIR, BAIDU_CACHE_DIR, RAW_DIR, FILTERED_DIR, BAIDU_DIR, FINAL_DIR, SRC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cookie配置（需定期更新）
# ⚠️ 重要：请将下面的 Cookie 替换为你自己的豆瓣登录 Cookie
# 获取方法：浏览器登录豆瓣后，打开开发者工具 → Application → Cookies → movie.douban.com
COOKIES = {
    "bid": "",
    "dbcl2": "",
    "ck": "",
    # 在此填入更多豆瓣 Cookie 键值对
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://movie.douban.com/chart",
    "Accept": "application/json, text/plain, */*",
}

# EDA分析配置
EDA_OUTPUT_DIR = BASE_DIR / "analysis" / "output"
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 300
PLOT_FONT_SIZE = 10
PLOT_CHINESE_FONT = "Microsoft YaHei"

# 确保EDA输出目录存在
EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)