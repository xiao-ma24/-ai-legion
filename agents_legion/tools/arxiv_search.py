"""ArXiv 论文搜索工具"""

import re
import time
import arxiv

_last_call = 0

# 中文 → 英文关键词映射
_KEYWORD_MAP = {
    '大语言模型': 'large language model', '大模型': 'large language model',
    '人工智能': 'artificial intelligence', '深度学习': 'deep learning',
    '机器学习': 'machine learning', '强化学习': 'reinforcement learning',
    '自然语言处理': 'natural language processing',
    '智能体': 'agent', '代理': 'agent', '框架': 'framework',
    '推理': 'reasoning', '规划': 'planning', '论文': 'paper',
    '竞赛': 'competition', '数学建模': 'mathematical modeling',
    '选题': 'topic selection', '分析': 'analysis',
    '神经网络': 'neural network', '卷积': 'convolution',
    '变换器': 'transformer', '注意力': 'attention',
    '知识图谱': 'knowledge graph', '图神经网络': 'graph neural network',
    '目标检测': 'object detection', '语义': 'semantic',
    '生成': 'generation', '对话': 'dialogue', '问答': 'question answering',
    '摘要': 'summarization', '翻译': 'translation',
    '多模态': 'multimodal', '视觉': 'vision', '图像': 'image',
    '推荐系统': 'recommendation system', '联邦学习': 'federated learning',
}


def _to_english_query(query: str) -> str:
    """将中文查询转换为英文关键词（ArXiv 不支持中文）"""
    # 如果已经是英文，直接返回
    if not re.search(r'[一-鿿]', query):
        return query

    keywords = []

    # 提取已有英文词
    en_words = re.findall(r'[a-zA-Z]{2,}', query)
    keywords.extend(en_words)

    # 中文关键词映射
    for cn, en in _KEYWORD_MAP.items():
        if cn in query and en not in keywords:
            keywords.append(en)

    if keywords:
        return ' '.join(keywords[:6])

    return 'artificial intelligence research'


async def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """搜索 ArXiv 论文，自带限流保护和中文关键词转换"""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 3:
        time.sleep(3 - elapsed)
    _last_call = time.time()

    query = _to_english_query(query)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors[:3]],
                "year": paper.published.year if paper.published else "",
                "url": paper.entry_id,
                "summary": paper.summary[:400].replace("\n", " "),
            })
        return results
    except Exception as e:
        return [{"title": "ArXiv 搜索暂不可用", "authors": [], "url": "", "summary": str(e)[:100]}]
