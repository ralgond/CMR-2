from collections import defaultdict

def weighted_rrf(doc_lists, weights, k=60):
    """
    计算加权倒数排序融合 (Weighted RRF) 分数并返回排序后的文档列表。
    
    Args:
        doc_lists: 二维列表，包含多个检索器返回的文档列表。每个子列表中的文档按相关性从高到低排列。
        weights: 列表，与 doc_lists 长度一致，表示每个检索器对应的权重。
        k: int, 排名常数（Rank Constant），用于平滑分数差异，通常默认设置为 60。
        
    Returns:
        list: 按照加权 RRF 分数降序排列的文档列表。
    """
    if len(doc_lists) != len(weights):
        raise ValueError("检索结果列表的数量必须与权重列表的数量相等。")
    
    # 使用字典存储每个文档的累计 RRF 分数
    rrf_scores = defaultdict(float)
    
    # 遍历每一个检索器的结果及其对应的权重
    for doc_list, weight in zip(doc_lists, weights):
        # enumerate 从 1 开始计数，符合排名从 1 开始的规则
        for rank, doc in enumerate(doc_list, start=1):
            # 核心公式: weight * [1 / (rank + k)]
            rrf_scores[doc] += weight * (1.0 / (rank + k))
            
    # 根据累计得分对文档进行降序排序
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    return sorted_docs

# --- 测试示例 ---
if __name__ == "__main__":
    # 模拟两路召回的结果
    bm25_results = ["Doc_A", "Doc_B", "Doc_C"]
    vector_results = ["Doc_B", "Doc_C", "Doc_A"]
    
    # 假设我们更信任向量检索，给向量检索分配更高的权重
    retriever_weights = [0.3, 0.7] 
    
    # 执行加权 RRF 融合
    final_ranking = weighted_rrf(
        doc_lists=[bm25_results, vector_results], 
        weights=retriever_weights
    )
    
    print("最终融合排序结果:", final_ranking)