"""CSV file loading utilities."""
from pathlib import Path
import pandas as pd
from ai_starter import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """CSV file loader utility class."""

    def load_files(self, data_dir: str | Path) -> dict[str, pd.DataFrame]:
        """Load all CSV files from a directory.

        Args:
            data_dir: Path to the directory containing CSV files.

        Returns:
            Dictionary mapping table names (file stems) to DataFrames.
        """
        data_dir = Path(data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
        data_dict = {}
        for csv_file in csv_files:
            table_name = csv_file.stem  # 使用文件名（不含扩展名）作为表名
            print(f"processing file: {csv_file} -> table: {table_name}")
            try:
                df = pd.read_csv(csv_file)
                data_dict[table_name] = df
            except Exception as e:
                logger.error(f"Error parsing {csv_file}: {str(e)}")
        return data_dict


if __name__ == "__main__":
    # 测试 CSV 加载功能 - 使用 RelationalTestData
    logger.info("=== Test: CSVLoader ===\n")

    loader = CSVLoader()
    data_dir = "../../data/RelationalTestData"

    logger.info(f"加载 CSV 文件: {data_dir}")
    data_dict = loader.load_files(data_dir)

    logger.info(f"\n✓ 加载了 {len(data_dict)} 个表:")
    for table_name, df in data_dict.items():
        logger.info(f"  表名: {table_name}")
        logger.info(f"    shape={df.shape}, columns={list(df.columns)}")
        logger.info(f"    前 3 行数据:")
        for row in df.head(3).itertuples(index=False):
            logger.info(f"      {row}")

    logger.info("\n✓ 测试完成")
