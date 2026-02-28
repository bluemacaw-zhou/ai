"""Table information generation utilities."""
from pathlib import Path
import json
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
import pandas as pd
from ai_starter import get_logger, ZhipuLLMFactory
from workflow_demo.model import TableInfo

logger = get_logger(__name__)


class TableInfoGenerator:
    """Table information generator utility class."""

    DEFAULT_PROMPT = """
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise.
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """

    def __init__(
        self,
        output_dir: str | Path = "WikiTableQuestions_TableInfo",
        prompt_str: str | None = None,
    ):
        """Initialize TableInfoGenerator.

        Args:
            output_dir: Directory to save table info JSON files.
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
        df: pd.DataFrame,
        llm: LLM,
        max_retries: int = 10,
    ) -> TableInfo:
        """Generate table info for a single DataFrame (internal method).

        Args:
            df: Input DataFrame to generate info for.
            llm: LLM instance for structured prediction.
            max_retries: Maximum number of retries if table name conflicts.

        Returns:
            Generated TableInfo object.

        Raises:
            RuntimeError: If unable to generate unique table name after max_retries.
        """
        for attempt in range(max_retries):
            df_str = df.head(10).to_csv()
            table_info = llm.structured_predict(
                TableInfo,
                self.prompt_tmpl,
                table_str=df_str,
                exclude_table_name_list=str(list(self.table_names)),
            )
            table_name = table_info.table_name

            if table_name not in self.table_names:
                self.table_names.add(table_name)
                print(f"Processed table: {table_name}")
                return table_info
            else:
                print(f"Table name {table_name} already exists, trying again (attempt {attempt + 1}/{max_retries}).")

        raise RuntimeError(f"Unable to generate unique table name after {max_retries} attempts")

    def process_dataframes(
        self,
        dfs: list[pd.DataFrame],
        llm: LLM | None = None,
        use_cache: bool = True,
    ) -> list[TableInfo]:
        """Process multiple DataFrames to generate table information.

        Args:
            dfs: List of DataFrames to process.
            llm: LLM instance for structured prediction. If None, creates one using ZhipuLLMFactory.
            use_cache: Whether to load cached table info if available.

        Returns:
            List of TableInfo objects corresponding to the input DataFrames.
        """
        # Auto-create LLM if not provided
        if llm is None:
            llm = ZhipuLLMFactory.create()

        table_infos = []

        for idx, df in enumerate(dfs):
            table_info = None

            # Try to load from cache if enabled
            if use_cache:
                table_info = self._load_cached_table_info(idx)

            # Generate if not cached
            if table_info is None:
                table_info = self._generate_table_info(df, llm)
                self._save_table_info(idx, table_info)

            table_infos.append(table_info)

        return table_infos


if __name__ == "__main__":
    # 测试 TableInfoGenerator 功能
    import pandas as pd

    logger.info("=== Test: TableInfoGenerator ===")
    generator = TableInfoGenerator(output_dir="workspace/table_info_test")

    # 创建测试数据
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

    # 从 config.yaml 读取配置
    try:
        logger.info("测试 process_dataframes() 方法（会保存文件）")
        table_infos = generator.process_dataframes(dfs, use_cache=False)

        logger.info(f"✓ 生成并保存了 {len(table_infos)} 个表信息")
        for i, info in enumerate(table_infos):
            logger.info(f"  [{i}] {info.table_name}: {info.table_summary}")

        logger.info(f"✓ 文件已保存到: {generator.output_dir}")

    except ValueError as e:
        logger.error(f"✗ Error: {e}")
        logger.error("Please configure zhipu.api_key in config.yaml")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        raise
