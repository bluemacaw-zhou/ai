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
    text,
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

        # 启用 WAL 模式支持并发读写
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False}
        )
        self.metadata_obj = MetaData()

        # 启用 WAL 模式（Write-Ahead Logging）
        if db_path != ":memory:":
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

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

        # Create the table in the database (if not exists)
        self.metadata_obj.create_all(self.engine)

        # Clear existing data from the table
        with self.engine.connect() as conn:
            conn.execute(table.delete())
            conn.commit()

        # Insert data from DataFrame into the table
        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_stmt = table.insert().values(**row.to_dict())
                conn.execute(insert_stmt)
            conn.commit()

    def load_dataframes(
        self,
        data_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Load multiple DataFrames into SQLite database.

        Args:
            data_dict: Dictionary mapping table names to DataFrames.
        """
        for table_name, df in data_dict.items():
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
    from pathlib import Path
    from workflow_demo.utils.csv_loader import CSVLoader

    logger.info("=== Test: SQLiteLoader ===\n")

    # 从 CSV 加载测试数据
    logger.info("从 CSV 加载测试数据: ../../data/RelationalTestData")
    csv_loader = CSVLoader()
    data_dict = csv_loader.load_files("../../data/RelationalTestData")

    logger.info(f"✓ 加载了 {len(data_dict)} 个表")
    logger.info(f"  表名: {list(data_dict.keys())}")

    # 设置数据库文件路径
    db_path = Path("../../db/workflow-demo.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 删除旧数据库（如果存在）
    if db_path.exists():
        db_path.unlink()
        logger.info(f"  删除旧数据库: {db_path}")

    # 加载到 SQLite
    logger.info(f"\n加载 {len(data_dict)} 个表到 SQLite: {db_path}")
    loader = SQLiteLoader(str(db_path))
    loader.load_dataframes(data_dict)
    logger.info(f"✓ 数据库创建成功")

    # 测试查询
    logger.info("\n验证数据:")
    with loader.engine.connect() as conn:
        for table_name in data_dict.keys():
            result = pd.read_sql_query(f"SELECT * FROM {table_name}", loader.engine)
            logger.info(f"  表 '{table_name}': {len(result)} 行, 列={list(result.columns)}")

    logger.info("\n✓ 测试完成")
