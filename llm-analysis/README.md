# 电影剧情智能分析系统 (LLM Plot Analysis)

## 项目简介

使用 DeepSeek 大模型 API 分析豆瓣电影的剧情简介，自动为电影打上语义标签（如"催泪"、"主旋律"、"合家欢"等），并分析不同档期的内容特征与电影表现之间的关系。

本项目是"计算思维期末大作业"的第3部分，实现了大模型 API 调用与智能分析功能。

## 功能特点

- 🤖 **批量智能分析**: 使用 DeepSeek API 批量处理电影剧情简介
- 🏷️ **语义标签生成**: 自动为电影打上预定义的语义标签
- 📊 **多维度可视化**: 生成标签分布、档期关联、性能分析、共现网络等4类图表
- 💰 **成本可控**: 支持成本预估、断点续传、小样本测试
- 📈 **洞察挖掘**: 分析不同档期的内容偏好与成功要素

## 项目结构

```
llm-analysis/
├── config.py                    # 配置文件（API设置、标签定义）
├── requirements.txt             # Python依赖
├── .env.example                 # API密钥模板
├── .gitignore                   # Git忽略规则
├── README.md                    # 本文档
├── src/
│   ├── 1_analyze_plots.py      # 主分析脚本
│   ├── 2_generate_insights.py  # 可视化生成脚本
│   └── utils.py                # 工具函数
├── data/
│   └── output/                 # 输出数据（标签结果、统计报告）
└── visualizations/             # 生成的图表
```

## 环境配置

### 1. 安装依赖

```bash
cd llm-analysis
pip install -r requirements.txt
```

**依赖说明**:
- `openai>=1.0.0` - DeepSeek 使用 OpenAI 兼容 API
- `pandas` - 数据处理
- `matplotlib`, `seaborn` - 可视化
- `tqdm` - 进度条显示
- `tenacity` - API 重试逻辑
- `networkx` - 网络图绘制

### 2. 申请 DeepSeek API Key

#### 步骤详解

**第一步：注册账号**
1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 点击右上角"注册"按钮
3. 使用邮箱或手机号完成注册

**第二步：实名认证（如需要）**
- 根据页面提示完成实名认证
- 部分功能可能需要认证后才能使用

**第三步：获取 API Key**
1. 登录后进入控制台
2. 找到"API 密钥"页面
3. 点击"创建新密钥"
4. 复制生成的密钥（⚠️ 只显示一次，请妥善保存！）

**第四步：充值（按需）**
- 进入"账户余额"页面
- 根据预估成本充值
- **DeepSeek 价格** (2026年):
  - 输入: ~¥1/百万tokens
  - 输出: ~¥2/百万tokens
- **成本估算**: 处理约6000部电影预计花费 **¥2-5** 元

**第五步：配置密钥**

```bash
# 1. 复制模板文件
cp .env.example .env

# 2. 编辑 .env 文件，填入你的API密钥
# Windows: notepad .env
# Mac/Linux: nano .env
```

在 `.env` 文件中填入：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 数据准备

确保 `../doban-data/data/final/china_all.xlsx` 文件存在。

该文件由 `doban-data` 项目生成，包含电影的剧情简介字段。

## 使用方法

### 步骤1: 成本估算 (推荐)

```bash
python src/1_analyze_plots.py --dry-run
```

**输出示例**:
```
============================================================
💰 成本估算
============================================================
待处理电影数量: 5,876
预计输入 Tokens: 2,350,400
预计输出 Tokens: 293,800

预计成本:
  输入: ¥2.35 CNY
  输出: ¥0.59 CNY
  ────────────────
  总计: ¥2.94 CNY (约 $0.41 USD)
============================================================
```

### 步骤2: 执行分析

#### 2.1 测试运行（推荐先测试）

```bash
python src/1_analyze_plots.py --limit 50
```

处理前50部电影，验证配置正确、标签合理。

#### 2.2 完整运行

```bash
python src/1_analyze_plots.py
```

**功能特性**:
- ✅ 自动批量处理所有电影（每批10部）
- ✅ 显示实时进度条
- ✅ 每100部保存检查点（支持断点续传）
- ✅ 自动重试失败的API调用
- ✅ 输出: `data/output/tagged_movies.csv`

**中断后恢复**:
```bash
# 程序会自动检测检查点，从上次中断处继续
python src/1_analyze_plots.py
```

**强制重新处理**:
```bash
python src/1_analyze_plots.py --force
```

### 步骤3: 生成可视化

```bash
python src/2_generate_insights.py
```

**生成内容**:

**数据文件** (`data/output/`):
- `tagged_movies.csv` - 包含标签的完整电影数据
- `tag_statistics.csv` - 标签频率和性能统计
- `schedule_tag_analysis.csv` - 档期-标签关联分析

**可视化图表** (`visualizations/`):
1. `tag_distribution.png` - 标签频率分布（横向柱状图）
2. `schedule_tag_heatmap.png` - 档期-标签热力图
3. `tag_performance.png` - 标签性能分析（散点图）
4. `tag_cooccurrence.png` - 标签共现网络图

## 预定义标签列表

本系统使用39个预定义语义标签，分为6大类：

| 类别 | 标签 |
|------|------|
| **情感** | 催泪、温馨、感动、虐心、治愈 |
| **风格** | 喜剧、黑色幽默、荒诞、文艺、商业 |
| **类型** | 动作、悬疑、惊悚、科幻、奇幻、爱情、犯罪 |
| **主题** | 主旋律、青春、励志、家庭、历史、战争、都市 |
| **受众** | 合家欢、成人向、青少年、儿童 |
| **节奏** | 烧脑、紧张刺激、轻松搞笑、沉重压抑 |

## 常见问题

### Q1: API Key 无效

**症状**: 提示 "Invalid API Key" 或 401 错误

**解决**:
- ✅ 检查 `.env` 文件中的密钥是否完整复制
- ✅ 确认账户余额充足
- ✅ 验证密钥未过期（DeepSeek平台查看）
- ✅ 确保没有多余空格或换行符

### Q2: 成本控制

**问题**: 担心费用过高

**建议**:
1. 先用 `--dry-run` 估算成本
2. 用 `--limit 50` 测试小样本
3. 修改 `config.py` 中的 `BATCH_SIZE` 调整速度
4. 支持断点续传，可随时暂停（Ctrl+C）

### Q3: 处理速度慢

**现象**: 6000部电影耗时较长

**说明**:
- DeepSeek API 有速率限制
- 批量处理会自动控制请求频率（每批延迟5秒）
- **预计耗时**: 30-60分钟（取决于网络和API响应）
- 可以后台运行，程序会自动保存进度

### Q4: 标签不准确

**现象**: 某些电影的标签不太合适

**说明**:
- LLM 分析有一定主观性
- 可在 `data/output/tagged_movies.csv` 中手动调整标签
- 可修改 `config.py` 中的预定义标签列表
- 可调整 `temperature` 参数（在 `1_analyze_plots.py` 中）

### Q5: 缺少中文字体

**症状**: 可视化图表中文显示为方块

**解决**:
- 程序会自动查找 Microsoft YaHei、SimHei 等字体
- 如仍有问题，手动指定字体路径（修改 `src/utils.py`）
- Windows 系统通常无此问题

### Q6: 数据文件找不到

**症状**: `FileNotFoundError: ../doban-data/data/final/china_all.xlsx`

**解决**:
- 确保先运行 `doban-data` 项目的数据采集流程
- 检查文件路径是否正确
- 可在 `config.py` 中修改 `DATA_SOURCE` 路径

## 技术架构

- **API**: DeepSeek Chat API (OpenAI-compatible)
- **数据处理**: Pandas
- **可视化**: Matplotlib, Seaborn, NetworkX
- **错误处理**: Tenacity (指数退避重试)
- **进度显示**: tqdm
- **配置管理**: python-dotenv

## 预期分析结果

运行完整流程后，可能发现：

1. **最常见标签**: 剧情、爱情、喜剧（因为这些是主流类型）
2. **春节档特征**: 合家欢、喜剧、动作标签占比高
3. **高表现标签**: "催泪"、"主旋律"可能与高评价人数相关
4. **标签组合**: "喜剧+合家欢"、"动作+商业"经常共现
5. **档期差异**: 文艺片在春节档表现可能不如其他档期

这些洞察对应作业要求3："分析哪种题材的简介在春节档最受欢迎"

## 命令行参数

### 1_analyze_plots.py

```bash
python src/1_analyze_plots.py [选项]

选项:
  --dry-run          仅估算成本，不实际调用API
  --limit N          仅处理前N部电影（用于测试）
  --force            强制重新处理，忽略检查点
```

### 2_generate_insights.py

```bash
python src/2_generate_insights.py [选项]

选项:
  --top-n N          可视化中显示前N个标签（默认20）
```

## 开发与扩展

### 修改标签列表

编辑 `config.py` 中的 `PREDEFINED_TAGS`:

```python
PREDEFINED_TAGS = {
    "情感": ["催泪", "温馨", "感动", "虐心", "治愈"],
    # 添加你自己的标签类别
    "新类别": ["标签1", "标签2", "标签3"],
}
```

### 修改批量大小

在 `config.py` 中调整：

```python
BATCH_SIZE = 10  # 改为5或20
```

- 更小 = 更快失败恢复，但API调用次数增加
- 更大 = 减少API调用，但单次失败影响更多电影

### 修改提示词

在 `src/1_analyze_plots.py` 中修改 `SYSTEM_PROMPT` 变量，自定义模型行为。

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 查看项目文档

---

**提示**: 首次使用建议先用 `--limit 10` 测试，确认一切正常后再运行完整流程。
