"""CSV file loading utilities."""
from pathlib import Path
import pandas as pd
from ai_starter import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """CSV file loader utility class."""

    def load_files(self, data_dir: str | Path) -> list[pd.DataFrame]:
        """Load all CSV files from a directory.

        Args:
            data_dir: Path to the directory containing CSV files.

        Returns:
            List of pandas DataFrames loaded from the CSV files.
        """
        data_dir = Path(data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
        dfs = []
        for csv_file in csv_files:
            print(f"processing file: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error parsing {csv_file}: {str(e)}")
        return dfs


if __name__ == "__main__":
    # 测试 CSV 加载功能
    loader = CSVLoader()
    data_dir = "../WikiTableQuestions/csv/200-csv"  # 修改为你的 CSV 文件目录
    dfs = loader.load_files(data_dir)
    logger.info(f"Loaded {len(dfs)} DataFrames")
    for i, df in enumerate(dfs):
        logger.info(f"DataFrame {i}: shape={df.shape}, columns={list(df.columns)}")
