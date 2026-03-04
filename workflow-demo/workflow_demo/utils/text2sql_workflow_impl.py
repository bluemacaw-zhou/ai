"""Text-to-SQL Workflow implementation."""
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from llama_index.core.llms import LLM
from ai_starter import get_logger

# 支持两种导入方式：相对导入（作为模块导入时）和绝对导入（直接运行时）
try:
    from .text2sql_index_builder import Text2SQLIndexBuilder
except ImportError:
    from workflow_demo.utils.text2sql_index_builder import Text2SQLIndexBuilder

try:
    from ..model import TableRetrieveEvent, TextToSQLEvent
except ImportError:
    from workflow_demo.model import TableRetrieveEvent, TextToSQLEvent

logger = get_logger(__name__)


class Text2SQLWorkflowImpl(Workflow):
    """Text-to-SQL Workflow implementation.

    This class defines the workflow steps for converting natural language
    queries to SQL and executing them.
    """

    def __init__(
        self,
        builder: Text2SQLIndexBuilder,
        llm: LLM,
        *args,
        **kwargs
    ) -> None:
        """Initialize workflow.

        Args:
            builder: Text2SQL index builder instance.
            llm: LLM instance for generating SQL and responses.
            *args, **kwargs: Additional arguments passed to Workflow base class.
        """
        super().__init__(*args, **kwargs)
        self._builder = builder
        self._llm = llm

    @property
    def builder(self) -> Text2SQLIndexBuilder:
        """Get Text2SQL index builder."""
        return self._builder

    @property
    def llm(self) -> LLM:
        """Get LLM instance."""
        return self._llm

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve relevant tables based on the query.

        Args:
            ctx: Workflow context.
            ev: Start event containing the user query.

        Returns:
            TableRetrieveEvent with retrieved table context.
        """
        table_schema_objs = self._builder.obj_retriever.retrieve(ev.query)
        table_context_str = self._builder.get_table_context_str(table_schema_objs)

        # 提取表名用于日志
        table_names = [obj.table_name for obj in table_schema_objs]
        logger.info(f"[步骤1] 检索相关表: {', '.join(table_names)}")
        logger.debug(f"表结构详情:\n{table_context_str}")

        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(
        self, ctx: Context, ev: TableRetrieveEvent
    ) -> TextToSQLEvent:
        """Generate SQL statement from natural language query.

        Args:
            ctx: Workflow context.
            ev: TableRetrieveEvent with table context and query.

        Returns:
            TextToSQLEvent with generated SQL.
        """
        fmt_messages = self._builder.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self._llm.chat(fmt_messages)
        sql = self._builder.parse_response_to_sql(chat_response)

        logger.info(f"[步骤2] 生成 SQL: {sql.replace(chr(10), ' ')}")  # 单行显示
        logger.debug(f"完整 SQL:\n{sql}")

        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Execute SQL and generate natural language response.

        Args:
            ctx: Workflow context.
            ev: TextToSQLEvent with SQL and query.

        Returns:
            StopEvent with the final response.
        """
        retrieved_rows = self._builder.sql_retriever.retrieve(ev.sql)

        # 提取实际查询结果
        if retrieved_rows and len(retrieved_rows) > 0:
            result_data = retrieved_rows[0].node.metadata.get('result', [])
            row_count = len(result_data) if isinstance(result_data, list) else 0
            logger.info(f"[步骤3] 执行 SQL 并获取结果: {row_count} 行数据")
            logger.debug(f"查询结果: {result_data}")
        else:
            logger.info("[步骤3] 执行 SQL: 无结果")

        fmt_messages = self._builder.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = self._llm.chat(fmt_messages)

        logger.info(f"[步骤4] 生成答案:\n{chat_response.message.content}")

        return StopEvent(result=chat_response)
