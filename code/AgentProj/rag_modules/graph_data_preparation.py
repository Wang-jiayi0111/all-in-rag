"""
图数据库数据准备模块
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]

class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取数据并转换为文档"""
    
    def __init__(self, data_path: str, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.recipes: List[GraphNode] = []
        self.ingredients: List[GraphNode] = []
        self.cooking_steps: List[GraphNode] = []

        self.data_path = data_path
        self.parent_child_map: Dict[str, str] = {}

        # 初始化文件映射字典
        self.file_map: Dict[str, Path] = {}
        # 建立本地文件索引
        if self.data_path:
            self._build_local_file_map()
        
        self._connect()
        
    def _build_local_file_map(self):
        """扫描本地目录，建立{菜名：文件路径} 的映射"""
        if not self.data_path:
            return
        
        logger.info(f"正在扫描本地文件映射：{self.data_path}")
        path_obj = Path(self.data_path)
        
        count = 0
        for file_path in path_obj.rglob("*.md"):
            # 假设文件名（不含扩展名）即为菜名
            dish_name = file_path.stem 
            self.file_map[dish_name] = file_path
            count += 1
        logger.info(f"已建立 {count} 个本地文件的索引映射")
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def load_graph_data(self) -> Dict[str, Any]:
        """
        从Neo4j加载图数据（抽取——使用 Cypher 语句抽取菜谱、食材、步骤的数据）
        
        Returns:
            包含节点（食材）和关系（步骤）的数据字典
        """
        logger.info("正在从Neo4j加载图数据...")
        
        with self.driver.session() as session:
            # 加载所有菜谱节点，从Category关系中读取分类信息
            recipes_query = """
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'
            OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WITH r, collect(c.name) as categories
            RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name, 
                   properties(r) as originalProperties,
                   CASE WHEN size(categories) > 0 
                        THEN categories[0] 
                        ELSE COALESCE(r.category, '未知') END as mainCategory,
                   CASE WHEN size(categories) > 0 
                        THEN categories 
                        ELSE [COALESCE(r.category, '未知')] END as allCategories
            ORDER BY r.nodeId
            """
            
            result = session.run(recipes_query)
            self.recipes = []
            for record in result:
                # 合并原始属性和新的分类信息
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]
                
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.recipes.append(node)
            
            logger.info(f"加载了 {len(self.recipes)} 个菜谱节点")
            
            # 加载所有食材节点
            ingredients_query = """
            MATCH (i:Ingredient)
            WHERE i.nodeId >= '200000000'
            RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
                   properties(i) as properties
            ORDER BY i.nodeId
            """
            
            result = session.run(ingredients_query)
            self.ingredients = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.ingredients.append(node)
            
            logger.info(f"加载了 {len(self.ingredients)} 个食材节点")
            
            # 加载所有烹饪步骤节点
            steps_query = """
            MATCH (s:CookingStep)
            WHERE s.nodeId >= '200000000'
            RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
                   properties(s) as properties
            ORDER BY s.nodeId
            """
            
            result = session.run(steps_query)
            self.cooking_steps = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.cooking_steps.append(node)
            
            logger.info(f"加载了 {len(self.cooking_steps)} 个烹饪步骤节点")
        
        return {
            'recipes': len(self.recipes),
            'ingredients': len(self.ingredients),
            'cooking_steps': len(self.cooking_steps)
        }
    
    def build_recipe_documents(self) -> List[Document]:
        """
        构建菜谱文档，集成相关的食材和步骤信息（转换——将load_graph_data提取的信息构建成Document类型）
        
        Returns:
            结构化的菜谱文档列表
        """
        logger.info("正在构建菜谱文档...")
        
        documents = []
        
        with self.driver.session() as session:
            for recipe in self.recipes:
                try:
                    recipe_id = recipe.node_id
                    recipe_name = recipe.name
                    
                    # 获取菜谱的相关食材
                    ingredients_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
                    RETURN i.name as name, i.category as category, 
                           req.amount as amount, req.unit as unit,
                           i.description as description
                    ORDER BY i.name
                    """
                    
                    ingredients_result = session.run(ingredients_query, {"recipe_id": recipe_id})
                    ingredients_list = list(ingredients_result)
                    ingredients_count = len(ingredients_list)

                    # 获取菜谱的烹饪步骤
                    steps_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
                    RETURN s.name
                    """
                    
                    steps_result = session.run(steps_query, {"recipe_id": recipe_id})
                    steps_list = list(steps_result)
                    steps_count = len(steps_list)

                    # -------------------------------------------------------
                    # 步骤 2: 构建 page_content 
                    # -------------------------------------------------------
                    full_content = ""
                    source_type = "neo4j_generated"

                    # 读取本地.md文件
                    if recipe_name in self.file_map:
                        try:
                            file_path = self.file_map[recipe_name]
                            with open(file_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                            source_type = "local_file_enhanced"
                        except Exception as e:
                            logger.warning(f"读取本地文件失败 {recipe_name}: {e}")
                    
                    # 用Neo4j拼接
                    if not full_content:
                        logger.warning(f"[{recipe_name}] 未找到本地文件，使用 Neo4j 数据拼接")
                        content_parts = [f"# {recipe_name}"]
                        if recipe.properties.get("description"):
                            content_parts.append(f"\n## 菜品描述\n{recipe.properties['description']}")
                        
                        # 简易拼接食材
                        if ingredients_list:
                            content_parts.append("\n## 所需食材")
                            for ing in ingredients_list:
                                content_parts.append(f"- {ing['name']}")
                                
                        full_content = "\n".join(content_parts)
                        source_type = "neo4j_fallback"
                    
                    # 创建文档对象
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            # --- 核心标识 ---
                            "node_id": recipe_id,
                            "recipe_name": recipe_name,
                            "node_type": "Recipe",
                            "content_source": source_type,

                            # --- 业务属性 ---
                            "category": recipe.properties.get("category", "未知"),
                            "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                            "difficulty": recipe.properties.get("difficulty", 0),
                            "doc_type": "recipe",

                            # --- 统计属性 ---
                            "ingredients_count": ingredients_count,
                            "steps_count": steps_count,

                            # --- 详情属性 ---
                            "prep_time": recipe.properties.get("prepTime", ""),
                            "cook_time": recipe.properties.get("cookTime", ""),
                            "servings": recipe.properties.get("servings", ""),
                            "content_length": len(full_content)
                        }
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"构建菜谱文档失败 {recipe_name} (ID: {recipe_id}): {e}")
                    continue
        
        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个菜谱文档")
        return documents


    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        对文档进行分块处理
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文档列表
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        
        if not self.documents:
            raise ValueError("请先构建document")
        
        chunks = []
        chunk_id = 0
        
        for doc in self.documents:
            content = doc.page_content
            
            # 简单的按长度分块
            if len(content) <= chunk_size:
                # 内容较短，不需要分块
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按章节分块（基于标题）
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))
                        
                        chunk_content = content[start:end]
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 按章节分块
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题
                            chunk_content = f"## {section}"
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                                "section_title": section.split('\n')[0] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_recipes': len(self.recipes),
            'total_ingredients': len(self.ingredients),
            'total_cooking_steps': len(self.cooking_steps),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }
        
        if self.documents:
            # 分类统计
            categories = {}
            cuisines = {}
            difficulties = {}
            
            for doc in self.documents:
                category = doc.metadata.get('category', '未知')
                categories[category] = categories.get(category, 0) + 1
                
                cuisine = doc.metadata.get('cuisine_type', '未知')
                cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
                
                difficulty = doc.metadata.get('difficulty', 0)
                difficulties[str(difficulty)] = difficulties.get(str(difficulty), 0) + 1
            
            stats.update({
                'categories': categories,
                'cuisines': cuisines,
                'difficulties': difficulties,
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
            })
        
        return stats
    
    def print_graph_stats(self):
        with self.driver.session() as session:
            # 查询节点
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            # 查询边
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            
            print(f"✅ 图谱统计完毕：")
            print(f"📊 总节点数 (Nodes): {node_count}")
            print(f"🔗 总边数 (Edges): {rel_count}")
    
    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 