"""
基于图数据库的RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""
    # 路径配置
    data_path: str = "./dataset_dishes"
    index_save_path: str = "./vector_index"

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-chat"

    # 重排设置
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    reranking_batch_size: int = 32

    # 融合重排配置
    enable_fusion_reranking: bool = True          # 启用融合重排
    fusion_semantic_weight: float = 0.6           # 语义权重
    fusion_graph_weight: float = 0.4              # 图路径权重
    enable_adaptive_weight: bool = True           # 根据查询复杂度动态调整
    enable_mmr: bool = False                      # 是否使用 MMR 消除重复
    mmr_lambda: float = 0.7 

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 