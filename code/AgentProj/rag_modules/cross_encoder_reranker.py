"""
Cross-Encoder 重排模块
用于在检索后重新排序文档，提高相关性
"""

import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-Encoder 重排器"""
    
    def __init__(self, model_name: str):
        """
        初始化 Cross-Encoder
        
        Args:
            model_name: 模型选择
        """
        try:
            logger.info(f"加载 Cross-Encoder 模型: {model_name}")
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            self.device = "cuda" if self._has_gpu() else "cpu"
        except Exception as e:
            logger.error(f"加载 Cross-Encoder 模型失败: {e}")
            raise
    
    def _has_gpu(self) -> bool:
        """检查是否有GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document],
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Document]:
        """
        对文档进行重排
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 返回的文档数
            batch_size: 批处理大小
        
        Returns:
            按相关性排序的文档列表
        """
        if not documents:
            logger.warning("没有文档需要重排")
            return []
        
        if len(documents) == 1:
            return documents
        
        try:
            logger.info(f"使用 Cross-Encoder 重排 {len(documents)} 个文档...")
            
            # 构建 query-document 对
            doc_texts = [doc.page_content for doc in documents]
            query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
            
            # 获取重排分数
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # 添加分数到文档元数据
            for doc, score in zip(documents, scores):
                doc.metadata['cross_encoder_score'] = float(score)
            
            # 按分数排序
            reranked_docs = sorted(
                documents,
                key=lambda x: x.metadata.get('cross_encoder_score', 0),
                reverse=True
            )
            
            # 返回 top_k
            logger.info(f"重排完成，返回 top-{min(top_k, len(reranked_docs))} 文档")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"重排过程出错: {e}")
            return documents[:top_k]
    
    def get_scores_with_texts(self, query: str, documents: List[Document]) -> List[Tuple[str, float]]:
        """
        获取文档和分数（用于调试）
        
        Returns:
            [(文档内容, 分数)] 列表
        """
        if not documents:
            return []
        
        doc_texts = [doc.page_content for doc in documents]
        query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
        scores = self.model.predict(query_doc_pairs)
        
        return list(zip(doc_texts, scores))