"""Text-to-SQL Workflow Demo.

Demonstrates the complete Text-to-SQL pipeline:
1. Load CSV files from WikiTableQuestions dataset
2. Generate table descriptions using LLM
3. Store tables in SQLite database
4. Build vector index for table retrieval
5. Run Text-to-SQL workflow to answer natural language queries
"""
import os
import asyncio
from pathlib import Path
from workflow_demo.utils import (
    CSVLoader,
    TableInfoGenerator,
    SQLiteLoader,
    Text2SQLIndexBuilder,
    Text2SQLWorkflowRunner,
)
from ai_starter import get_logger, ZhipuGlobalSettings

logger = get_logger(__name__)


class WikiTableQAPipeline:
    """WikiTableQuestions Q&A Pipeline.

    Complete pipeline for loading WikiTableQuestions data,
    building indexes, and answering natural language queries.
    """

    def __init__(
        self,
        data_dir: str = "./WikiTableQuestions/csv/200-csv",
        db_path: str = "db/wiki_table_questions.db",
        table_info_dir: str = "db/table_info/wiki_table_questions",
        similarity_top_k: int = 3,
        use_cache: bool = True,
    ):
        """Initialize WikiTableQA pipeline.

        Args:
            data_dir: Directory containing CSV files.
            db_path: Path to SQLite database file.
            table_info_dir: Directory for storing table info cache.
            similarity_top_k: Number of similar tables to retrieve.
            use_cache: Whether to use cached table info.
        """
        self._data_dir = Path(data_dir)
        self._db_path = db_path
        self._table_info_dir = table_info_dir
        self._similarity_top_k = similarity_top_k
        self._use_cache = use_cache

        # Components (lazy initialization)
        self._llm = None
        self._embed_model = None
        self._dfs = None
        self._table_infos = None
        self._engine = None
        self._builder = None
        self._workflow_runner = None

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self._data_dir

    @property
    def db_path(self) -> str:
        """Get database path."""
        return self._db_path

    @property
    def table_info_dir(self) -> str:
        """Get table info directory."""
        return self._table_info_dir

    @property
    def llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            self._setup_models()
        return self._llm

    @property
    def embed_model(self):
        """Get embedding model instance (lazy initialization)."""
        if self._embed_model is None:
            self._setup_models()
        return self._embed_model

    @property
    def dfs(self) -> dict:
        """Get loaded DataFrames dict (lazy initialization)."""
        if self._dfs is None:
            self._load_data()
        return self._dfs

    @property
    def table_infos(self):
        """Get table info list (lazy initialization)."""
        if self._table_infos is None:
            self._generate_table_infos()
        return self._table_infos

    @property
    def engine(self):
        """Get SQLAlchemy engine (lazy initialization)."""
        if self._engine is None:
            self._load_to_database()
        return self._engine

    @property
    def builder(self) -> Text2SQLIndexBuilder:
        """Get Text2SQL index builder (lazy initialization)."""
        if self._builder is None:
            self._build_index()
        return self._builder

    @property
    def workflow_runner(self) -> Text2SQLWorkflowRunner:
        """Get workflow runner (lazy initialization)."""
        if self._workflow_runner is None:
            self._create_workflow()
        return self._workflow_runner

    def _setup_models(self) -> None:
        """Setup LLM and Embedding models."""
        logger.info("Setting up LLM and Embedding models (from config.yaml)")
        self._llm, self._embed_model = ZhipuGlobalSettings.setup()
        logger.info("Models setup completed")

    def _load_data(self) -> None:
        """Load CSV files."""
        logger.info(f"Loading CSV files from {self._data_dir}")
        loader = CSVLoader()
        self._dfs = loader.load_files(str(self._data_dir))
        logger.info(f"Loaded {len(self._dfs)} tables: {list(self._dfs.keys())}")

    def _generate_table_infos(self) -> None:
        """Generate table descriptions."""
        logger.info("Generating table descriptions")
        table_info_generator = TableInfoGenerator(output_dir=self._table_info_dir)
        self._table_infos = table_info_generator.process_dataframes(
            self.dfs, llm=self.llm, use_cache=self._use_cache
        )
        logger.info(f"Generated {len(self._table_infos)} table descriptions")

    def _load_to_database(self) -> None:
        """Load data to SQLite database."""
        logger.info(f"Loading data to SQLite: {self._db_path}")
        sqlite_loader = SQLiteLoader(self._db_path)
        sqlite_loader.load_dataframes(self.dfs)
        self._engine = sqlite_loader.get_engine()
        logger.info("Data loaded to database")

    def _build_index(self) -> None:
        """Build vector index for table retrieval."""
        logger.info("Building vector index")
        self._builder = Text2SQLIndexBuilder(
            engine=self.engine,
            table_infos=self.table_infos,
            similarity_top_k=self._similarity_top_k,
        )
        logger.info("Vector index built")

    def _create_workflow(self) -> None:
        """Create workflow runner."""
        logger.info("Creating workflow runner")
        self._workflow_runner = Text2SQLWorkflowRunner(
            builder=self.builder,
            llm=self.llm,
            verbose=True,
        )
        logger.info("Workflow runner created")

    def print_text2sql_prompt(self) -> None:
        """Print text2sql prompt for debugging."""
        self.builder.print_text2sql_prompt()

    async def query(self, question: str):
        """Query the pipeline with a natural language question.

        Args:
            question: Natural language question.

        Returns:
            Response from the workflow.
        """
        logger.info(f"Processing query: {question}")
        response = await self.workflow_runner.run(question)
        logger.info("Query completed")
        return response

    def visualize_workflow(self, filename: str = "text_to_sql_workflow.html") -> None:
        """Visualize the workflow diagram.

        Args:
            filename: Output HTML filename.
        """
        logger.info(f"Generating workflow diagram: {filename}")
        self.workflow_runner.visualize(filename)


async def main():
    """Main entry point - Text-to-SQL JOIN 查询完整测试."""
    logger.info("=" * 70)
    logger.info("Text-to-SQL JOIN 查询完整测试")
    logger.info("=" * 70)

    # Initialize pipeline with RelationalTestData
    pipeline = WikiTableQAPipeline(
        data_dir="../data/RelationalTestData",
        db_path="../db/workflow-demo.db",
        table_info_dir="../db/table_info/relational",
        similarity_top_k=3,
        use_cache=True,
    )

    # 强制完成所有初始化（触发懒加载）
    logger.info("\n初始化 Pipeline...")
    _ = pipeline.workflow_runner  # 触发所有懒加载
    logger.info("✓ Pipeline 初始化完成\n")

    # 测试不同复杂度的查询（覆盖所有场景）
    test_queries = [
        # 基础单表查询
        "What are the names of all users?",
        "How many products are there?",
        "List all product names and their prices",

        # 两表 JOIN
        "Show me all orders with user names",
        "Which users have placed orders?",

        # 三表 JOIN
        "List all orders with user names and product names",
        "Show order details including buyer name and product name",

        # 聚合查询
        "How many orders did each user place?",
        "What is the average age of users?",

        # 复杂查询（聚合 + JOIN + 计算）
        "What is the total amount spent by each user? (quantity * price)",
        "Which product has been ordered the most?",
    ]

    logger.info(f"\n准备测试 {len(test_queries)} 个查询\n")

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[测试 {i}/{len(test_queries)}] {query}")
        logger.info('='*70)

        response = await pipeline.query(query)

    logger.info(f"\n{'='*70}")
    logger.info("✓ 所有测试完成")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
