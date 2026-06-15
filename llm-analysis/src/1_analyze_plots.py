# -*- coding: utf-8 -*-
"""
步骤1: LLM 剧情分析

功能:
1. 加载电影数据
2. 筛选有效剧情简介
3. 批量调用 DeepSeek API 分析剧情
4. 为每部电影打上语义标签
5. 保存结果
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.utils import load_api_key, estimate_cost, print_cost_estimate, validate_tags, truncate_summary

import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# 系统提示词
SYSTEM_PROMPT = f"""你是一个专业的电影内容分析助手。根据电影的剧情简介，为每部电影打上1-3个最贴切的语义标签。

可选标签列表：
{', '.join(config.ALL_TAGS)}

规则：
1. 严格从上述标签列表中选择，不要创造新标签
2. 每部电影选择1-3个最符合的标签
3. 优先选择最能体现电影核心特征的标签
4. 只返回JSON格式，不要其他解释"""


def load_movie_data():
    """加载电影数据"""
    print("\n" + "="*60)
    print("步骤 1.1: 加载电影数据")
    print("="*60)

    if not config.DATA_SOURCE.exists():
        raise FileNotFoundError(f"数据文件不存在: {config.DATA_SOURCE}")

    df = pd.read_excel(config.DATA_SOURCE)
    print(f"加载数据: {config.DATA_SOURCE}")
    print(f"总记录数: {len(df)}")

    return df


def filter_valid_movies(df):
    """筛选有效的电影记录"""
    print("\n" + "="*60)
    print("步骤 1.2: 筛选有效剧情简介")
    print("="*60)

    initial_count = len(df)

    # 筛选有剧情简介的记录
    df_valid = df[df['剧情简介'].notna()].copy()
    print(f"有剧情简介: {len(df_valid)}/{initial_count}")

    # 筛选简介长度 >= MIN_SUMMARY_LENGTH
    df_valid = df_valid[df_valid['剧情简介'].str.len() >= config.MIN_SUMMARY_LENGTH].copy()
    print(f"简介长度 >= {config.MIN_SUMMARY_LENGTH}: {len(df_valid)}/{initial_count}")

    # 移除"暂无简介"等无效简介
    invalid_keywords = ['暂无简介', '无简介', '待更新']
    for keyword in invalid_keywords:
        df_valid = df_valid[~df_valid['剧情简介'].str.contains(keyword, na=False)]

    print(f"最终有效记录: {len(df_valid)}/{initial_count} ({len(df_valid)/initial_count*100:.1f}%)")

    return df_valid


def build_batch_prompt(movies_batch):
    """构建批量分析提示词

    Args:
        movies_batch: 电影批次 DataFrame

    Returns:
        str: 构建的提示词
    """
    movie_list = []
    for idx, row in movies_batch.iterrows():
        title = row['电影名']
        summary = truncate_summary(row['剧情简介'], max_length=500)
        movie_list.append(f"{idx}. 【{title}】\n{summary}")

    movies_text = "\n\n".join(movie_list)

    user_prompt = f"""请分析以下{len(movies_batch)}部电影的剧情简介，为每部打上1-3个语义标签：

{movies_text}

请以JSON格式返回（使用电影序号作为key）：
{{
  "序号1": ["标签1", "标签2"],
  "序号2": ["标签1"],
  ...
}}

只返回JSON，不要其他解释。"""

    return user_prompt


@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_deepseek_api(client, user_prompt):
    """调用 DeepSeek API

    Args:
        client: OpenAI client
        user_prompt: 用户提示词

    Returns:
        dict: 解析后的JSON响应
    """
    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # 强制JSON输出
            temperature=0.3,  # 降低温度以获得更一致的结果
            timeout=config.REQUEST_TIMEOUT
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  JSON解析失败: {e}")
        print(f"响应内容: {content[:200]}...")
        raise
    except Exception as e:
        print(f"⚠️  API调用失败: {e}")
        raise


def save_checkpoint(results, checkpoint_num):
    """保存检查点

    Args:
        results: 结果字典
        checkpoint_num: 检查点编号
    """
    checkpoint_file = config.OUTPUT_DIR / f"checkpoint_{checkpoint_num:04d}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_checkpoints():
    """加载所有检查点

    Returns:
        dict: 合并的结果字典
    """
    checkpoint_files = sorted(config.OUTPUT_DIR.glob("checkpoint_*.json"))
    if not checkpoint_files:
        return {}

    print(f"\n发现 {len(checkpoint_files)} 个检查点文件")

    all_results = {}
    for checkpoint_file in checkpoint_files:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            all_results.update(results)

    print(f"已加载 {len(all_results)} 条记录")
    return all_results


def process_movies(df_valid, dry_run=False, limit=None, force=False):
    """处理电影数据

    Args:
        df_valid: 有效电影 DataFrame
        dry_run: 是否仅估算成本
        limit: 限制处理数量（用于测试）
        force: 是否强制重新处理（忽略检查点）

    Returns:
        dict: 电影标签字典 {subject_id: [tags]}
    """
    print("\n" + "="*60)
    print("步骤 1.3: 处理电影剧情")
    print("="*60)

    # 限制处理数量
    if limit:
        df_valid = df_valid.head(limit)
        print(f"⚠️  测试模式：仅处理前 {limit} 部电影")

    # 成本估算
    print_cost_estimate(len(df_valid))

    if dry_run:
        print("✓ 这是一次试运行，未进行实际API调用")
        return {}

    # 加载检查点（如果存在且非强制模式）
    processed_results = {} if force else load_checkpoints()

    # 筛选未处理的电影
    if processed_results and not force:
        processed_ids = set(processed_results.keys())
        df_to_process = df_valid[~df_valid['subject_id'].astype(str).isin(processed_ids)]
        print(f"已处理: {len(processed_results)} 部")
        print(f"待处理: {len(df_to_process)} 部")
    else:
        df_to_process = df_valid
        print(f"待处理: {len(df_to_process)} 部")

    if len(df_to_process) == 0:
        print("\n✓ 所有电影已处理完成")
        return processed_results

    # 加载API密钥
    api_key = load_api_key()

    # 分批处理
    results = processed_results.copy()
    batches = [(batch_idx, df_to_process.iloc[i:i+config.BATCH_SIZE])
               for batch_idx, i in enumerate(range(0, len(df_to_process), config.BATCH_SIZE))]

    results_lock = threading.Lock()
    checkpoint_counter = len(processed_results) // 100

    def process_one_batch(batch_idx, batch):
        """处理单个批次（在单独线程中运行）"""
        # 每个线程独立创建 API 客户端（线程安全）
        thread_client = OpenAI(
            api_key=api_key,
            base_url=config.DEEPSEEK_API_BASE_URL
        )
        user_prompt = build_batch_prompt(batch)
        api_response = call_deepseek_api(thread_client, user_prompt)

        batch_results = {}
        for idx, row in batch.iterrows():
            subject_id = str(row['subject_id'])
            title = row['电影名']
            tags = None
            for key in [str(idx), title]:
                if key in api_response:
                    tags = api_response[key]
                    break
            if tags is None:
                tags = []
            valid_tags = validate_tags(tags, title)
            batch_results[subject_id] = valid_tags
        return batch_results

    print(f"\n开始并发处理 {len(batches)} 个批次（{config.MAX_CONCURRENT_BATCHES} 线程）...")

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_BATCHES) as executor:
        future_to_batch = {
            executor.submit(process_one_batch, batch_idx, batch): batch_idx
            for batch_idx, batch in batches
        }

        completed_count = 0
        for future in tqdm(as_completed(future_to_batch), total=len(batches),
                           desc="处理批次", unit="batch"):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                with results_lock:
                    results.update(batch_results)
                    completed_count += len(batch_results)
                    # 每100部电影保存一次检查点（加锁保护）
                    if len(results) // 100 > checkpoint_counter:
                        checkpoint_counter = len(results) // 100
                        save_checkpoint(results, checkpoint_counter)
            except Exception as e:
                print(f"\n⚠️  批次 {batch_idx+1} 处理失败（其他批次继续）: {e}")
                with results_lock:
                    save_checkpoint(results, checkpoint_counter)

    # 保存最终检查点
    save_checkpoint(results, checkpoint_counter + 1)

    return results


def save_results(df_original, results):
    """保存结果

    Args:
        df_original: 原始 DataFrame
        results: 标签结果字典
    """
    print("\n" + "="*60)
    print("步骤 1.4: 保存结果")
    print("="*60)

    # 添加标签列
    df_output = df_original.copy()
    df_output['LLM标签'] = df_output['subject_id'].astype(str).map(
        lambda x: ', '.join(results.get(x, []))
    )

    # 保存为CSV
    output_file = config.OUTPUT_DIR / "tagged_movies.csv"
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✓ 结果已保存: {output_file}")
    print(f"  总记录数: {len(df_output)}")
    print(f"  已打标签: {df_output['LLM标签'].notna().sum()}")

    # 打印标签统计
    print("\n标签统计:")
    all_tags_list = []
    for tags_str in df_output['LLM标签'].dropna():
        if tags_str:
            all_tags_list.extend([t.strip() for t in tags_str.split(',')])

    from collections import Counter
    tag_counts = Counter(all_tags_list)
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: {count} 次")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM 电影剧情分析")
    parser.add_argument('--dry-run', action='store_true', help='仅估算成本，不实际调用API')
    parser.add_argument('--limit', type=int, help='限制处理数量（用于测试）')
    parser.add_argument('--force', action='store_true', help='强制重新处理，忽略检查点')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM 电影剧情分析系统")
    print("="*60)

    # 1. 加载数据
    df = load_movie_data()

    # 2. 筛选有效记录
    df_valid = filter_valid_movies(df)

    # 3. 处理电影
    results = process_movies(df_valid, dry_run=args.dry_run, limit=args.limit, force=args.force)

    # 4. 保存结果
    if not args.dry_run and results:
        save_results(df, results)

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
