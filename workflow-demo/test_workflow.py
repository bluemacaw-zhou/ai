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
        db_path: str = "wiki_table_questions.db",
        table_info_dir: str = "WikiTableQuestions_TableInfo",
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
    def dfs(self):
        """Get loaded DataFrames (lazy initialization)."""
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
        logger.info(f"Loaded {len(self._dfs)} CSV files")

    def _generate_table_infos(self) -> None:
        """Generate table descriptions."""
        logger.info("Generating table descriptions")
        table_info_generator = TableInfoGenerator(output_dir=self._table_info_dir)
        self._table_infos = table_info_generator.process_dataframes(
            self.dfs, self.llm, use_cache=self._use_cache
        )
        logger.info(f"Generated {len(self._table_infos)} table descriptions")

    def _load_to_database(self) -> None:
        """Load data to SQLite database."""
        logger.info(f"Loading data to SQLite: {self._db_path}")
        sqlite_loader = SQLiteLoader(self._db_path)
        table_names = [info.table_name for info in self.table_infos]
        sqlite_loader.load_dataframes(self.dfs, table_names)
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
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("WikiTableQuestions Text-to-SQL Demo")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = WikiTableQAPipeline(
        data_dir="./WikiTableQuestions/csv/200-csv",
        db_path="wiki_table_questions.db",
        table_info_dir="WikiTableQuestions_TableInfo",
        similarity_top_k=3,
        use_cache=True,
    )

    # Print text2sql prompt for debugging
    pipeline.print_text2sql_prompt()

    # Query the pipeline
    question = "What was the year that The Notorious B.I.G was signed to Bad Boy?"
    response = await pipeline.query(question)

    logger.info("=" * 60)
    logger.info("Query Result")
    logger.info("=" * 60)
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {response}")

    # Visualize workflow
    pipeline.visualize_workflow("text_to_sql_workflow.html")

    logger.info("=" * 60)
    logger.info("Demo completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
