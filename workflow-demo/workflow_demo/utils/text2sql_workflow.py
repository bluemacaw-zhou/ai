"""Text-to-SQL Workflow utilities."""
from llama_index.core.llms import LLM
from ai_starter import get_logger, ZhipuLLMFactory

# 支持两种导入方式：相对导入（作为模块导入时）和绝对导入（直接运行时）
from workflow_demo.utils.text2sql_workflow_impl import Text2SQLWorkflowImpl
from workflow_demo.utils.text2sql_index_builder import Text2SQLIndexBuilder

logger = get_logger(__name__)


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
    def workflow(self) -> Text2SQLWorkflowImpl:
        """Get workflow instance (lazy initialization)."""
        if self._workflow is None:
            self._workflow = Text2SQLWorkflowImpl(
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

        Note: This method is currently disabled as draw_all_possible_flows
        is not available in the current llama-index version.

        Args:
            filename: Output HTML filename. Defaults to "text_to_sql_workflow.html".
        """
        # TODO: Find alternative for workflow visualization
        # from llama_index.utils.workflow import draw_all_possible_flows
        # draw_all_possible_flows(Text2SQLWorkflowImpl, filename=filename)
        logger.warning(f"Workflow visualization is currently disabled: {filename}")
        # print(f"Workflow diagram saved to: {filename}")


if __name__ == "__main__":
    # 测试 Text2SQLWorkflowRunner 功能
    import asyncio
    from pathlib import Path

    async def test_workflow():
        from workflow_demo.utils.table_info_generator import TableInfoGenerator
        from sqlalchemy import create_engine

        logger.info("=== Test: Create and run Text2SQLWorkflow ===\n")

        db_path = Path("../../db/workflow-demo.db")
        table_info_dir = Path("../../db/table_info/relational")

        # 检查数据库是否存在
        if not db_path.exists():
            logger.error(f"✗ 数据库不存在: {db_path}")
            logger.error("请先运行以下命令构建数据库:")
            logger.error("  python csv_loader.py")
            logger.error("  python table_info_generator.py")
            logger.error("  python sqlite_loader.py")
            return

        # 检查 TableInfo 是否存在
        if not table_info_dir.exists() or not list(table_info_dir.glob("*.json")):
            logger.error(f"✗ TableInfo 不存在: {table_info_dir}")
            logger.error("请先运行: python table_info_generator.py")
            return

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

        logger.info(f"✓ 加载了 {len(table_infos)} 个 TableInfo\n")

        # 创建数据库连接
        engine = create_engine(f"sqlite:///{db_path}")

        try:
            # 创建 Text2SQLIndexBuilder
            logger.info("创建 Text2SQLIndexBuilder...")
            builder = Text2SQLIndexBuilder(
                engine=engine,
                table_infos=table_infos,
                auto_setup_models=True
            )
            logger.info("✓ Text2SQLIndexBuilder 创建成功\n")

            # 创建 Text2SQLWorkflowRunner
            logger.info("创建 Text2SQLWorkflowRunner...")
            runner = Text2SQLWorkflowRunner(builder)
            logger.info("✓ Text2SQLWorkflowRunner 创建成功\n")

            # 运行简单查询测试
            query = "What are the names of all users?"
            logger.info(f"测试查询: {query}")
            await runner.run(query)

            logger.info("\n✓ 测试完成")

        except ValueError as e:
            logger.error(f"✗ Error: {e}")
            logger.error("Please configure zhipu.api_key in config.yaml")
        except Exception as e:
            logger.error(f"✗ Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    asyncio.run(test_workflow())
