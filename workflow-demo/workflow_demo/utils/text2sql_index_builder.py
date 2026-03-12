"""Text-to-SQL index and retriever builder utilities."""
from typing import List
from sqlalchemy import Engine
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex, PromptTemplate
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.llms import ChatResponse
from ai_starter import get_logger
from ai_starter.llama_index import ZhipuGlobalSettings

# 支持两种导入方式：相对导入（作为模块导入时）和绝对导入（直接运行时）
try:
    from ..model import TableInfo
except ImportError:
    from workflow_demo.model import TableInfo

logger = get_logger(__name__)


class Text2SQLIndexBuilder:
    """Text-to-SQL index and retriever builder.

    Automatically configures Settings.llm and Settings.embed_model if not already set.
    """

    def __init__(
        self,
        engine: Engine,
        table_infos: List[TableInfo],
        similarity_top_k: int = 3,
        auto_setup_models: bool = True,
    ):
        """Initialize Text2SQLIndexBuilder.

        Args:
            engine: SQLAlchemy engine instance.
            table_infos: List of TableInfo objects with table metadata.
            similarity_top_k: Number of similar tables to retrieve. Defaults to 3.
            auto_setup_models: If True and Settings are not configured, automatically
                               setup LLM and Embedding using ZhipuGlobalSettings. Defaults to True.
        """
        self.engine = engine
        self.table_infos = table_infos
        self.similarity_top_k = similarity_top_k

        # Auto-setup models if needed
        if auto_setup_models:
            logger.info("配置 LLM 和 Embedding 模型...")
            self._ensure_models_configured()
            logger.info("✓ 模型配置完成")

        # Build core components
        logger.info("创建 SQL 数据库包装器...")
        self._sql_database = SQLDatabase(engine)

        logger.info("构建向量索引（需要调用 Embedding API）...")
        self._obj_retriever = self._build_object_retriever()
        logger.info("✓ 向量索引构建完成")

        logger.info("构建 SQL 检索器...")
        self._sql_retriever = self._build_sql_retriever()

        logger.info("构建提示词模板...")
        self._text2sql_prompt = self._build_text2sql_prompt()
        self._response_synthesis_prompt = self._build_response_synthesis_prompt()
        logger.info("✓ 所有组件构建完成")

    def _ensure_models_configured(self) -> None:
        """Ensure Settings.llm and Settings.embed_model are configured for ZhipuAI.

        Since this class is designed for ZhipuAI models, we always configure them.
        Only skip if they are already configured with ZhipuAI models.
        """
        # 检查是否已经配置了智谱模型（避免重复配置）
        already_configured = (
            Settings.llm is not None and
            hasattr(Settings.llm, 'model') and
            'glm' in Settings.llm.model  # 智谱模型名包含 'glm'
        )

        if already_configured:
            logger.debug("智谱AI 模型已配置，跳过")
            return

        logger.debug("配置智谱AI 模型...")
        ZhipuGlobalSettings.setup()

    @property
    def sql_database(self) -> SQLDatabase:
        """Get SQL database instance."""
        return self._sql_database

    @property
    def obj_retriever(self):
        """Get object retriever for table schema retrieval."""
        return self._obj_retriever

    @property
    def sql_retriever(self) -> SQLRetriever:
        """Get SQL retriever for executing SQL queries."""
        return self._sql_retriever

    @property
    def text2sql_prompt(self) -> PromptTemplate:
        """Get text-to-SQL prompt template."""
        return self._text2sql_prompt

    @property
    def response_synthesis_prompt(self) -> PromptTemplate:
        """Get response synthesis prompt template."""
        return self._response_synthesis_prompt

    def _build_object_retriever(self):
        """Build object retriever for table schemas.

        注意：当前向量索引存储在内存中（未持久化），程序重启后需要重新构建。

        向量索引结构示意图：
        ┌─────────────────────────────────────────────────────────────┐
        │ VectorStoreIndex (内存 SimpleVectorStore)                   │
        ├─────────────────────────────────────────────────────────────┤
        │ Node 1:                                                     │
        │   - table_name: "users"                                     │
        │   - context_str: "User information table containing..."     │
        │   - embedding: [0.123, 0.456, 0.789, ...]  (768维向量)      │
        ├─────────────────────────────────────────────────────────────┤
        │ Node 2:                                                     │
        │   - table_name: "products"                                  │
        │   - context_str: "Product catalog with product details..."  │
        │   - embedding: [0.234, 0.567, 0.890, ...]  (768维向量)      │
        └─────────────────────────────────────────────────────────────┘

        查询流程：
        用户问题 "What are the names of all users?"
            ↓ (调用 Embedding API)
        问题向量 [0.111, 0.444, 0.777, ...]
            ↓ (余弦相似度计算)
        找到最相关的 top_k 个表（如: users 相似度 0.92）
            ↓
        返回表的元数据（表名、表结构、表描述）

        Returns:
            Object retriever configured with vector index.
        """
        logger.debug("  - 创建表节点映射...")
        table_node_mapping = SQLTableNodeMapping(self._sql_database)

        logger.debug(f"  - 创建 {len(self.table_infos)} 个表 schema 对象...")
        # 构建被向量化的对象：每个表的元数据（表名 + 表描述）
        table_schema_objs = [
            SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
            for t in self.table_infos
        ]

        logger.debug("  - 调用 Embedding API 生成向量（可能需要几秒钟）...")
        # 构建向量索引（调用 Settings.embed_model.get_text_embedding()）
        # ⚠️ 未配置 StorageContext，向量数据存储在内存中（程序重启后丢失）
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,  # 默认使用内存中的 SimpleVectorStore
        )
        logger.debug("  - 创建检索器...")
        return obj_index.as_retriever(similarity_top_k=self.similarity_top_k)

    def _build_sql_retriever(self) -> SQLRetriever:
        """Build SQL retriever.

        Returns:
            SQL retriever instance.
        """
        return SQLRetriever(self._sql_database)

    def _build_text2sql_prompt(self) -> PromptTemplate:
        """Build text-to-SQL prompt template.

        Returns:
            Prompt template for text-to-SQL generation.
        """
        return DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
            dialect=self.engine.dialect.name
        )

    def _build_response_synthesis_prompt(self) -> PromptTemplate:
        """Build response synthesis prompt template.

        Returns:
            Prompt template for response synthesis.
        """
        response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the query results.\n"
            "Query: {query_str}\n"
            "SQL: {sql_query}\n"
            "SQL Response: {context_str}\n"
            "Response: "
        )
        return PromptTemplate(response_synthesis_prompt_str)

    @staticmethod
    def parse_response_to_sql(chat_response: ChatResponse) -> str:
        """Parse LLM response to extract SQL query.

        处理多种 LLM 响应格式：
        1. SQLQuery: SELECT ...
        2. ```sql\nSELECT ...```
        3. SELECT ...

        Args:
            chat_response: Chat response from LLM.

        Returns:
            Extracted and cleaned SQL query string.
        """
        response = chat_response.message.content

        # 1. 提取 "SQLQuery:" 之后的内容
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]

        # 2. 去除 "SQLResult:" 之后的内容
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]

        # 3. 去除 markdown 代码块标记
        response = response.strip().strip("```").strip()

        # 4. 去除代码块语言标识符（如 "sql", "sqlite"）
        # LLM 可能返回 "sql\nSELECT ..." 格式
        first_line = response.split('\n', 1)[0].strip().lower()
        if first_line in ('sql', 'sqlite', 'mysql', 'postgresql', 'postgres'):
            # 去除第一行
            response = response.split('\n', 1)[1] if '\n' in response else response

        return response.strip()

    def get_table_context_str(self, table_schema_objs: List[SQLTableSchema]) -> str:
        """Get table context string from schema objects.

        Args:
            table_schema_objs: List of SQLTableSchema objects.

        Returns:
            Formatted table context string with descriptions.
        """
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def print_text2sql_prompt(self) -> None:
        """Print the text-to-SQL prompt template for debugging."""
        print(self.text2sql_prompt.template)


if __name__ == "__main__":
    # 测试 Text2SQLIndexBuilder 功能
    from pathlib import Path
    from workflow_demo.utils.table_info_generator import TableInfoGenerator
    from sqlalchemy import create_engine

    logger.info("=== Test: Text2SQLIndexBuilder ===\n")

    db_path = Path("../../db/workflow-demo.db")
    table_info_dir = Path("../../db/table_info/relational")

    # 检查数据库是否存在
    if not db_path.exists():
        logger.error(f"✗ 数据库不存在: {db_path}")
        logger.error("请先运行以下命令构建数据库:")
        logger.error("  python csv_loader.py")
        logger.error("  python table_info_generator.py")
        logger.error("  python sqlite_loader.py")
        exit(1)

    # 检查 TableInfo 是否存在
    if not table_info_dir.exists() or not list(table_info_dir.glob("*.json")):
        logger.error(f"✗ TableInfo 不存在: {table_info_dir}")
        logger.error("请先运行: python table_info_generator.py")
        exit(1)

    # 加载数据
    logger.info(f"✓ 数据库: {db_path}")
    logger.info(f"✓ TableInfo 目录: {table_info_dir}\n")

    # 从缓存加载 TableInfo
    generator = TableInfoGenerator(output_dir=str(table_info_dir))
    table_infos = []
    for json_file in sorted(table_info_dir.glob("*.json")):
        idx = int(json_file.stem.split('_')[0])
        table_info = generator._load_cached_table_info(idx)
        if table_info:
            table_infos.append(table_info)

    logger.info(f"✓ 加载了 {len(table_infos)} 个 TableInfo")
    for info in table_infos:
        logger.info(f"  - {info.table_name}: {info.table_summary}")

    # 创建数据库连接
    engine = create_engine(f"sqlite:///{db_path}")

    # 构建 Text2SQL 索引
    logger.info("\n构建 Text2SQL 索引...")
    try:
        builder = Text2SQLIndexBuilder(
            engine=engine,
            table_infos=table_infos,
            similarity_top_k=3,
            auto_setup_models=True
        )
        logger.info("✓ Text2SQLIndexBuilder 创建成功")
        logger.info(f"  - SQL database: {builder.sql_database}")
        logger.info(f"  - Object retriever: {builder.obj_retriever}")
        logger.info(f"  - SQL retriever: {builder.sql_retriever}")
        logger.info("\n✓ 测试完成")
    except ValueError as e:
        logger.error(f"✗ Error: {e}")
        logger.error("Please configure zhipu.api_key in config.yaml")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        raise
