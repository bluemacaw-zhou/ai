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
from ai_starter import get_logger, ZhipuGlobalSettings
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
        """Ensure Settings.llm and Settings.embed_model are configured.

        If not configured, automatically setup using ZhipuGlobalSettings.

        Note: Settings.llm and Settings.embed_model have default OpenAI values,
        so we check if they are still using default models to determine if setup is needed.
        """
        # Check if using default OpenAI models (not configured for ZhipuAI)
        is_default_llm = (
            Settings.llm is None or
            (hasattr(Settings.llm, 'model') and Settings.llm.model == 'gpt-3.5-turbo')
        )
        is_default_embed = (
            Settings.embed_model is None or
            (hasattr(Settings.embed_model, 'model_name') and
             Settings.embed_model.model_name == 'text-embedding-ada-002')
        )

        if is_default_llm or is_default_embed:
            logger.debug("检测到默认 OpenAI 模型配置，切换到智谱AI模型")
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

        Returns:
            Object retriever configured with vector index.
        """
        logger.debug("  - 创建表节点映射...")
        table_node_mapping = SQLTableNodeMapping(self._sql_database)

        logger.debug(f"  - 创建 {len(self.table_infos)} 个表 schema 对象...")
        table_schema_objs = [
            SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
            for t in self.table_infos
        ]

        logger.debug("  - 调用 Embedding API 生成向量（可能需要几秒钟）...")
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
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

        Args:
            chat_response: Chat response from LLM.

        Returns:
            Extracted and cleaned SQL query string.
        """
        response = chat_response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip().strip("```").strip()

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
    import pandas as pd
    from workflow_demo.utils.sqlite_loader import SQLiteLoader

    logger.info("=== Test: Text2SQLIndexBuilder ===")

    # 1. 创建测试数据（与其他测试用例保持一致）
    dfs = [
        pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        }),
        pd.DataFrame({
            "product_id": [101, 102, 103],
            "product_name": ["Laptop", "Mouse", "Keyboard"],
            "price": [999.99, 29.99, 59.99]
        })
    ]

    # 2. 先创建 SQLite 表（表必须真实存在）
    logger.info("步骤 1: 创建 SQLite 表")
    loader = SQLiteLoader(":memory:")
    table_names = ["users", "products"]
    loader.load_dataframes(dfs, table_names)
    engine = loader.get_engine()
    logger.info(f"✓ 创建了 {len(table_names)} 个表")

    # 3. 创建 TableInfo 元数据
    logger.info("步骤 2: 创建表元数据")
    table_infos = [
        TableInfo(
            table_name="users",
            table_summary="User information table containing id, name, and age"
        ),
        TableInfo(
            table_name="products",
            table_summary="Product catalog with product details and prices"
        )
    ]

    # 4. 创建 Text2SQLIndexBuilder（会自动配置 LLM 和 Embedding）
    logger.info("步骤 3: 构建 Text2SQL 索引")
    try:
        builder = Text2SQLIndexBuilder(
            engine=engine,
            table_infos=table_infos,
            similarity_top_k=2,
            auto_setup_models=True
        )
        logger.info("✓ Text2SQLIndexBuilder 创建成功")
        logger.info(f"  - SQL database: {builder.sql_database}")
        logger.info(f"  - Object retriever: {builder.obj_retriever}")
        logger.info(f"  - SQL retriever: {builder.sql_retriever}")
        logger.info("✓ 测试完成")
    except ValueError as e:
        logger.error(f"✗ Error: {e}")
        logger.error("Please configure zhipu.api_key in config.yaml")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        raise
