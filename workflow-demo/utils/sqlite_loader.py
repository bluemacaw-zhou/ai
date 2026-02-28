"""SQLite database loading utilities."""
from pathlib import Path
import re
import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Engine,
)
from ai_starter import get_logger

logger = get_logger(__name__)


class SQLiteLoader:
    """SQLite database loader utility class."""

    def __init__(self, db_path: str | Path = ":memory:"):
        """Initialize SQLiteLoader.

        Args:
            db_path: Path to the SQLite database file.
                     Use ":memory:" for in-memory database.
        """
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            self.db_path = Path(db_path)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata_obj = MetaData()

    @staticmethod
    def _sanitize_column_name(col_name: str) -> str:
        """Remove special characters and replace spaces with underscores.

        Args:
            col_name: Original column name.

        Returns:
            Sanitized column name.
        """
        return re.sub(r"\W+", "_", col_name)

    def _create_table_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Create a table from a DataFrame using SQLAlchemy.

        Args:
            df: Input DataFrame.
            table_name: Name of the table to create.
        """
        # Sanitize column names
        sanitized_columns = {
            col: self._sanitize_column_name(col) for col in df.columns
        }
        df = df.rename(columns=sanitized_columns)

        # Dynamically create columns based on DataFrame columns and data types
        columns = [
            Column(col, String if dtype == "object" else Integer)
            for col, dtype in zip(df.columns, df.dtypes)
        ]

        # Create a table with the defined columns
        table = Table(table_name, self.metadata_obj, *columns)

        # Create the table in the database
        self.metadata_obj.create_all(self.engine)

        # Insert data from DataFrame into the table
        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_stmt = table.insert().values(**row.to_dict())
                conn.execute(insert_stmt)
            conn.commit()

    def load_dataframes(
        self,
        dfs: list[pd.DataFrame],
        table_names: list[str],
    ) -> None:
        """Load multiple DataFrames into SQLite database.

        Args:
            dfs: List of DataFrames to load.
            table_names: List of table names corresponding to the DataFrames.

        Raises:
            ValueError: If the number of DataFrames and table names don't match.
        """
        if len(dfs) != len(table_names):
            raise ValueError(
                f"Number of DataFrames ({len(dfs)}) must match "
                f"number of table names ({len(table_names)})"
            )

        for df, table_name in zip(dfs, table_names):
            print(f"Creating table: {table_name}")
            try:
                self._create_table_from_dataframe(df, table_name)
            except Exception as e:
                print(f"Error creating table {table_name}: {str(e)}")

    def get_engine(self) -> Engine:
        """Get the SQLAlchemy engine instance.

        Returns:
            SQLAlchemy Engine instance.
        """
        return self.engine


if __name__ == "__main__":
    # 测试 SQLite 加载功能
    import pandas as pd

    logger.info("=== Test: SQLiteLoader ===")
    loader = SQLiteLoader(":memory:")

    # 创建测试数据（与 table_info_generator.py 保持一致）
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
    table_names = ["users", "products"]

    logger.info(f"加载 {len(dfs)} 个 DataFrame 到 SQLite")
    loader.load_dataframes(dfs, table_names)
    logger.info(f"✓ 数据库创建成功: {loader.db_path}")

    # 测试查询
    logger.info("测试查询:")
    with loader.engine.connect() as conn:
        for table_name in table_names:
            result = pd.read_sql_query(f"SELECT * FROM {table_name}", loader.engine)
            logger.info(f"  表 '{table_name}' 有 {len(result)} 行数据")
            logger.info(f"    列: {list(result.columns)}")

    logger.info("✓ 测试完成")
