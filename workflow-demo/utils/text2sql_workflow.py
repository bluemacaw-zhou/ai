"""Text-to-SQL Workflow utilities."""
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event,
)
from llama_index.core.llms import LLM
from ai_starter import get_logger, ZhipuLLMFactory
from .text2sql_index_builder import Text2SQLIndexBuilder

logger = get_logger(__name__)


# 事件：找到数据库中相关的表
class TableRetrieveEvent(Event):
    """Result of running table retrieval."""

    table_context_str: str
    query: str


# 事件：文本转 SQL
class TextToSQLEvent(Event):
    """Text-to-SQL event."""

    sql: str
    query: str


class _Text2SQLWorkflowImpl(Workflow):
    """Internal Text-to-SQL Workflow implementation.

    This is the actual workflow class that extends Workflow.
    Use Text2SQLWorkflowRunner to create and run this workflow.
    """

    def __init__(
        self,
        builder: Text2SQLIndexBuilder,
        llm: LLM,
        *args,
        **kwargs
    ) -> None:
        """Init params."""
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
        """Retrieve tables."""
        table_schema_objs = self._builder.obj_retriever.retrieve(ev.query)
        table_context_str = self._builder.get_table_context_str(table_schema_objs)
        print("====\n" + table_context_str + "\n====")
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(
        self, ctx: Context, ev: TableRetrieveEvent
    ) -> TextToSQLEvent:
        """Generate SQL statement."""
        fmt_messages = self._builder.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self._llm.chat(fmt_messages)
        sql = self._builder.parse_response_to_sql(chat_response)
        print("====\n" + sql + "\n====")
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Run SQL retrieval and generate response."""
        retrieved_rows = self._builder.sql_retriever.retrieve(ev.sql)
        print("====\n" + str(retrieved_rows) + "\n====")
        fmt_messages = self._builder.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = self._llm.chat(fmt_messages)
        return StopEvent(result=chat_response)


class Text2SQLWorkflowRunner:
    """Text-to-SQL Workflow runner.

    Manages workflow creation and execution.
    """

    def __init__(
        self,
        builder: Text2SQLIndexBuilder,
        llm: LLM | None = None,
        verbose: bool = True,
    ):
        """Initialize workflow runner.

        Args:
            builder: Text2SQL index builder instance.
            llm: LLM instance. If None, creates one using ZhipuLLMFactory.
            verbose: Whether to print verbose workflow logs. Defaults to True.
        """
        self._builder = builder
        self._llm = llm or ZhipuLLMFactory.create()
        self._verbose = verbose
        self._workflow = None

    @property
    def builder(self) -> Text2SQLIndexBuilder:
        """Get Text2SQL index builder."""
        return self._builder

    @property
    def llm(self) -> LLM:
        """Get LLM instance."""
        return self._llm

    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self._verbose

    @property
    def workflow(self) -> _Text2SQLWorkflowImpl:
        """Get workflow instance (lazy initialization)."""
        if self._workflow is None:
            self._workflow = _Text2SQLWorkflowImpl(
                self._builder,
                self._llm,
                verbose=self._verbose,
            )
        return self._workflow

    async def run(self, query: str):
        """Run workflow with a query.

        Args:
            query: Natural language query.

        Returns:
            Workflow response containing the answer.
        """
        response = await self.workflow.run(query=query)
        return response

    def visualize(self, filename: str = "text_to_sql_workflow.html") -> None:
        """Visualize workflow diagram.

        Args:
            filename: Output HTML filename. Defaults to "text_to_sql_workflow.html".
        """
        from llama_index.utils.workflow import draw_all_possible_flows

        draw_all_possible_flows(_Text2SQLWorkflowImpl, filename=filename)
        print(f"Workflow diagram saved to: {filename}")


if __name__ == "__main__":
    # 测试 Text2SQLWorkflowRunner 功能
    import asyncio

    async def test_workflow():
        from sqlalchemy import create_engine
        from workflow_demo.model import TableInfo

        logger.info("=== Test: Create and run Text2SQLWorkflow ===")

        # 创建测试数据库和表
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    age INTEGER
                )
            """)
            conn.execute("INSERT INTO users VALUES (1, 'Alice', 25)")
            conn.execute("INSERT INTO users VALUES (2, 'Bob', 30)")
            conn.commit()

        # 创建表信息
        table_infos = [
            TableInfo(
                table_name="users",
                table_summary="User information table containing id, name, and age"
            )
        ]

        try:
            # 创建 builder 和 runner
            builder = Text2SQLIndexBuilder(
                engine=engine,
                table_infos=table_infos,
                auto_setup_models=True
            )
            runner = Text2SQLWorkflowRunner(builder)

            # 运行查询
            query = "What are the names of all users?"
            logger.info(f"Query: {query}")
            response = await runner.run(query)
            logger.info(f"Response: {response}")

            # 生成工作流图
            runner.visualize("workspace/text2sql_workflow.html")

        except ValueError as e:
            logger.error(f"Error: {e}")
            logger.error("Please set DASHSCOPE_API_KEY environment variable")

    asyncio.run(test_workflow())
