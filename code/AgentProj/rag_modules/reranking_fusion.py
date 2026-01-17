"""
多维度融合重排模块
融合 Cross-Encoder 语义相关性 + 图路径置信度
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class FusionRerankerConfig:
    """融合重排配置"""
    # 权重配置
    semantic_weight: float = 0.6      # Cross-Encoder 语义相关性权重
    graph_weight: float = 0.4         # 图路径置信度权重
    
    # 动态权重调整
    enable_adaptive_weight: bool = True  # 根据查询类型动态调整权重
    
    # 其他参数
    use_mmr: bool = False              # 是否使用最大边际相关性（MMR）消除重复
    mmr_lambda: float = 0.7            # MMR 参数：0 = 多样性，1 = 相关性
    
    # 正则化参数
    normalize_scores: bool = True      # 是否对分数进行 min-max 正则化

class FusionReranker:
    """多维度融合重排器"""
    
    def __init__(self, config: FusionRerankerConfig):
        self.config = config
        
    def fuse_and_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        query_complexity: float = 0.5  # 查询复杂度 0-1
    ) -> List[Document]:
        """
        融合 Cross-Encoder 和图路径置信度进行重排
        
        Args:
            query: 用户查询
            documents: 带有两种相关性分数的文档列表
            top_k: 返回的文档数
            query_complexity: 查询复杂度（0-简单，1-复杂）
        
        Returns:
            融合排序后的文档列表
        """
        if not documents:
            return []
        
        logger.info(f"执行多维度融合重排，共 {len(documents)} 个文档")
        
        # 1. 提取两种相关性分数
        documents_with_scores = self._extract_scores(documents)
        
        # 2. 正则化分数（可选）
        if self.config.normalize_scores:
            documents_with_scores = self._normalize_scores(documents_with_scores)
        
        # 3. 动态调整权重（可选）
        weights = self._get_dynamic_weights(query_complexity)
        
        # 4. 计算融合分数
        documents_with_scores = self._compute_fusion_scores(
            documents_with_scores,
            weights
        )
        
        # 5. 可选：使用 MMR 消除重复
        if self.config.use_mmr:
            logger.info("应用 MMR 消除重复...")
            documents_with_scores = self._apply_mmr(documents_with_scores, top_k)
        
        # 6. 按融合分数排序
        sorted_docs = sorted(
            documents_with_scores,
            key=lambda x: x['fusion_score'],
            reverse=True
        )
        
        # 7. 同步回文档元数据并返回
        result_docs = self._sync_metadata_and_return(sorted_docs, top_k)
        
        logger.info(f"融合重排完成，返回 top-{len(result_docs)} 文档")
        return result_docs
    
    def _extract_scores(self, documents: List[Document]) -> List[Dict]:
        """提取两种相关性分数"""
        docs_with_scores = []
        
        for doc in documents:
            semantic_score = doc.metadata.get('cross_encoder_score', 0.5)
            graph_score = doc.metadata.get('relevance_score', 0.5)  # 来自 GraphPath
            
            # 如果没有图分数，尝试从其他字段获取
            if graph_score == 0.5:
                graph_score = doc.metadata.get('path_relevance', 0.5)
            
            docs_with_scores.append({
                'doc': doc,
                'semantic_score': float(semantic_score),
                'graph_score': float(graph_score),
                'fusion_score': 0.0  # 稍后计算
            })
        
        logger.debug(f"提取分数完成: 语义分数范围 {min(d['semantic_score'] for d in docs_with_scores):.3f}-"
                    f"{max(d['semantic_score'] for d in docs_with_scores):.3f}, "
                    f"图分数范围 {min(d['graph_score'] for d in docs_with_scores):.3f}-"
                    f"{max(d['graph_score'] for d in docs_with_scores):.3f}")
        
        return docs_with_scores
    
    def _normalize_scores(self, docs_with_scores: List[Dict]) -> List[Dict]:
        """使用 min-max 正则化分数到 [0, 1] 范围"""
        
        # 分别对两种分数进行正则化
        semantic_scores = [d['semantic_score'] for d in docs_with_scores]
        graph_scores = [d['graph_score'] for d in docs_with_scores]
        
        semantic_min, semantic_max = min(semantic_scores), max(semantic_scores)
        graph_min, graph_max = min(graph_scores), max(graph_scores)
        
        for doc_dict in docs_with_scores:
            # 正则化语义分数
            if semantic_max > semantic_min:
                doc_dict['semantic_score_norm'] = (
                    (doc_dict['semantic_score'] - semantic_min) / (semantic_max - semantic_min)
                )
            else:
                doc_dict['semantic_score_norm'] = 0.5
            
            # 正则化图分数
            if graph_max > graph_min:
                doc_dict['graph_score_norm'] = (
                    (doc_dict['graph_score'] - graph_min) / (graph_max - graph_min)
                )
            else:
                doc_dict['graph_score_norm'] = 0.5
        
        logger.debug("分数正则化完成")
        return docs_with_scores
    
    def _get_dynamic_weights(self, query_complexity: float) -> Dict[str, float]:
        """
        根据查询复杂度动态调整权重
        
        简单查询：更依赖语义相关性
        复杂查询：更依赖图结构信息
        """
        if not self.config.enable_adaptive_weight:
            return {
                'semantic': self.config.semantic_weight,
                'graph': self.config.graph_weight
            }
        
        # 根据复杂度调整权重
        # 复杂度高时，增加图权重；复杂度低时，增加语义权重
        semantic_w = self.config.semantic_weight * (1 - query_complexity * 0.3)
        graph_w = self.config.graph_weight * (1 + query_complexity * 0.3)
        
        # 归一化
        total = semantic_w + graph_w
        weights = {
            'semantic': semantic_w / total,
            'graph': graph_w / total
        }
        
        logger.info(f"动态权重调整 (复杂度={query_complexity:.2f}): "
                   f"语义={weights['semantic']:.3f}, 图={weights['graph']:.3f}")
        
        return weights
    
    def _compute_fusion_scores(
        self,
        docs_with_scores: List[Dict],
        weights: Dict[str, float]
    ) -> List[Dict]:
        """计算融合分数"""
        
        for doc_dict in docs_with_scores:
            # 获取正则化后的分数，如果没有则使用原始分数
            semantic_score = doc_dict.get('semantic_score_norm', doc_dict['semantic_score'])
            graph_score = doc_dict.get('graph_score_norm', doc_dict['graph_score'])
            
            # 加权融合
            fusion_score = (
                weights['semantic'] * semantic_score +
                weights['graph'] * graph_score
            )
            
            doc_dict['fusion_score'] = fusion_score
            
            logger.debug(
                f"融合评分: {doc_dict['doc'].metadata.get('recipe_name', 'Unknown')}, "
                f"语义={semantic_score:.3f}, 图={graph_score:.3f}, "
                f"融合={fusion_score:.3f}"
            )
        
        return docs_with_scores
    
    def _apply_mmr(self, docs_with_scores: List[Dict], top_k: int) -> List[Dict]:
        """
        应用 MMR (Maximum Marginal Relevance) 消除相似文档重复
        
        选择相关性高但与已选文档差异性大的文档
        """
        if len(docs_with_scores) <= top_k:
            return docs_with_scores
        
        selected = []
        remaining = sorted(
            docs_with_scores,
            key=lambda x: x['fusion_score'],
            reverse=True
        )
        
        # 选择最相关的第一个文档
        selected.append(remaining.pop(0))
        
        # 迭代选择
        while len(selected) < top_k and remaining:
            # 计算每个候选的 MMR 分数
            mmr_scores = []
            
            for candidate in remaining:
                # 相关性部分
                relevance = candidate['fusion_score']
                
                # 与已选文档的相似度（这里使用简化的相似度）
                max_similarity = max(
                    self._document_similarity(
                        candidate['doc'],
                        selected_item['doc']
                    )
                    for selected_item in selected
                )
                
                # MMR 分数 = λ * 相关性 - (1-λ) * 最大相似度
                mmr = (
                    self.config.mmr_lambda * relevance -
                    (1 - self.config.mmr_lambda) * max_similarity
                )
                
                mmr_scores.append((candidate, mmr))
            
            # 选择 MMR 分数最高的
            best = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best)
            remaining.remove(best)
        
        logger.debug(f"MMR 完成: 从 {len(docs_with_scores)} 选出 {len(selected)} 个文档")
        return selected
    
    def _document_similarity(self, doc1: Document, doc2: Document) -> float:
        """简单的文档相似度计算（基于内容长度和元数据）"""
        # 这是一个简化实现，实际可以使用向量相似度
        content1 = doc1.page_content[:200]
        content2 = doc2.page_content[:200]
        
        # 计算字符串相似度（Jaccard）
        set1 = set(content1)
        set2 = set(content2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _sync_metadata_and_return(self, sorted_docs: List[Dict], top_k: int) -> List[Document]:
        """同步融合分数到文档元数据并返回"""
        result = []
        
        for i, doc_dict in enumerate(sorted_docs[:top_k], 1):
            doc = doc_dict['doc']
            
            # 更新元数据
            doc.metadata.update({
                'cross_encoder_score': doc_dict['semantic_score'],
                'graph_relevance_score': doc_dict['graph_score'],
                'fusion_score': doc_dict['fusion_score'],
                'fusion_rank': i
            })
            
            result.append(doc)
        
        return result