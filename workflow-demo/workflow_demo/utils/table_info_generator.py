"""Table information generation utilities."""
from pathlib import Path
import json
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
import pandas as pd
from ai_starter import get_logger
from ai_starter.llama_index import ZhipuLLMFactory
from workflow_demo.model import TableInfo

logger = get_logger(__name__)


class TableInfoGenerator:
    """Table information generator utility class."""

    DEFAULT_PROMPT = """
Analyze the table data and generate a concise summary with the following information:

**IMPORTANT**: The table name is: {table_name}

Please analyze this table and provide:
1. **Primary key**: Identify which column(s) could serve as the primary key (unique identifier, often named like "id", "user_id", "order_id", etc.)
2. **Foreign keys**: Identify columns that look like references to other tables.
   - Columns ending with "_id" (like "user_id", "product_id") usually indicate foreign keys
   - For example: "user_id" likely references a "users" table, "product_id" likely references a "products" table
3. **Brief description**: Describe what kind of data this table contains

Table data (first 10 rows):
{table_str}

**Example summary formats:**
- "users table. Primary key: user_id. Contains user demographic information."
- "orders table. Primary key: order_id. Foreign keys: user_id (likely references users table), product_id (likely references products table). Contains order transaction details."
- "products table. Primary key: product_id. Contains product catalog information with names and prices."

**Important**: Make sure to explicitly mention ALL foreign key columns you find (columns that end with "_id" or reference other tables).

Summary: """

    def __init__(
        self,
        output_dir: str | Path = "db/table_info",
        prompt_str: str | None = None,
    ):
        """Initialize TableInfoGenerator.

        Args:
            output_dir: Directory to save table info JSON files. Defaults to "db/table_info".
            prompt_str: Custom prompt string for table summary generation.
                       If None, uses the default prompt.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prompt = prompt_str or self.DEFAULT_PROMPT
        self.prompt_tmpl = ChatPromptTemplate(
            message_templates=[ChatMessage.from_str(prompt, role="user")]
        )

        self.table_names = set()

    def _load_cached_table_info(self, idx: int) -> TableInfo | None:
        """Load cached table info from file if exists.

        Args:
            idx: Index of the table.

        Returns:
            TableInfo object if cached file exists, None otherwise.

        Raises:
            ValueError: If more than one file matches the index.
        """
        results_gen = self.output_dir.glob(f"{idx}_*")
        results_list = list(results_gen)

        if len(results_list) == 0:
            return None
        elif len(results_list) == 1:
            path = results_list[0]
            with open(path, 'r') as file:
                data = json.load(file)
                return TableInfo.model_validate(data)
        else:
            raise ValueError(
                f"More than one file matching index: {results_list}"
            )

    def _save_table_info(self, idx: int, table_info: TableInfo) -> None:
        """Save table info to JSON file.

        Args:
            idx: Index of the table.
            table_info: TableInfo object to save.
        """
        out_file = self.output_dir / f"{idx}_{table_info.table_name}.json"
        with open(out_file, "w") as f:
            json.dump(table_info.model_dump(), f)

    def _generate_table_info(
        self,
        table_name: str,
        df: pd.DataFrame,
        llm: LLM,
    ) -> TableInfo:
        """Generate table info for a single DataFrame (internal method).

        Args:
            table_name: Name of the table (from CSV filename).
            df: Input DataFrame to generate info for.
            llm: LLM instance for structured prediction.

        Returns:
            Generated TableInfo object.
        """
        df_str = df.head(10).to_csv()
        table_info = llm.structured_predict(
            TableInfo,
            self.prompt_tmpl,
            table_name=table_name,
            table_str=df_str,
        )
        # 强制使用指定的表名
        table_info.table_name = table_name
        print(f"Processed table: {table_name}")
        return table_info

    def process_dataframes(
        self,
        data_dict: dict[str, pd.DataFrame],
        llm: LLM | None = None,
        use_cache: bool = True,
    ) -> list[TableInfo]:
        """Process multiple DataFrames to generate table information.

        Args:
            data_dict: Dictionary mapping table names to DataFrames.
            llm: LLM instance for structured prediction. If None, creates one using ZhipuLLMFactory.
            use_cache: Whether to load cached table info if available.

        Returns:
            List of TableInfo objects corresponding to the input DataFrames.
        """
        # Auto-create LLM if not provided
        if llm is None:
            llm = ZhipuLLMFactory.create()

        table_infos = []

        for idx, (table_name, df) in enumerate(data_dict.items()):
            table_info = None

            # Try to load from cache if enabled
            if use_cache:
                table_info = self._load_cached_table_info(idx)

            # Generate if not cached
            if table_info is None:
                table_info = self._generate_table_info(table_name, df, llm)
                self._save_table_info(idx, table_info)

            table_infos.append(table_info)

        return table_infos


if __name__ == "__main__":
    # 测试 TableInfoGenerator 功能
    from pathlib import Path
    from workflow_demo.utils.csv_loader import CSVLoader

    logger.info("=== Test: TableInfoGenerator ===\n")

    # 从 CSV 加载测试数据
    logger.info("从 CSV 加载测试数据: ../../data/RelationalTestData")
    csv_loader = CSVLoader()
    data_dict = csv_loader.load_files("../../data/RelationalTestData")
    logger.info(f"✓ 加载了 {len(data_dict)} 个表: {list(data_dict.keys())}")

    # 使用 LLM 生成 TableInfo（增强的 prompt 会推断主键/外键）
    logger.info("\n使用 LLM 生成 TableInfo（会推断主键/外键关系）")
    generator = TableInfoGenerator(output_dir="../../db/table_info/relational")

    try:
        table_infos = generator.process_dataframes(data_dict, use_cache=False)

        logger.info(f"\n✓ 生成并保存了 {len(table_infos)} 个表信息:")
        for i, info in enumerate(table_infos):
            logger.info(f"  [{i}] {info.table_name}")
            logger.info(f"      {info.table_summary}")

        logger.info(f"\n✓ 文件已保存到: {generator.output_dir}")
        logger.info("\n✓ 测试完成")

    except ValueError as e:
        logger.error(f"✗ Error: {e}")
        logger.error("Please configure zhipu.api_key in config.yaml")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
