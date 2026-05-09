"""Text-to-SQL Function Calling 演示

演示如何用 ZhipuAI GLM 实现“自然语言问题 -> SQL -> 查询数据库 -> 最终回答”：
- 模型根据表结构和用户问题生成 SQLite SQL
- Python 本地执行 query_db 工具
- 再把查询结果交给模型组织成自然语言答案
"""

import json
import sqlite3

from ai_starter import get_logger
from ai_starter.llm import ZhipuAIBase

logger = get_logger(__name__)


class SqlFunctionCallDemo:
    """使用 Function Calling 查询 SQLite 业务数据库。"""

    _database_schema = """
CREATE TABLE orders (
    id INT PRIMARY KEY NOT NULL, -- 主键，不允许为空
    customer_id INT NOT NULL, -- 客户ID，不允许为空
    product_id STR NOT NULL, -- 产品ID，不允许为空
    price DECIMAL(10,2) NOT NULL, -- 价格，不允许为空
    status INT NOT NULL, -- 订单状态。0代表待支付，1代表已支付，2代表已退款
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 创建时间
    pay_time TIMESTAMP -- 支付时间，可以为空
);
"""

    _tools = [
        {
            "type": "function",
            "function": {
                "name": "query_db",
                "description": "使用此函数查询业务数据库获取结果，SQL 必须能在 Python sqlite3 中执行。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "SQL query extracting info to answer the user's question. "
                                "The query should be returned in plain text, not in JSON. "
                                "The query should only contain grammars supported by SQLite."
                            ),
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def __init__(self) -> None:
        self._llm = ZhipuAIBase()
        self._conn = sqlite3.connect(":memory:")
        self._init_database()

    def close(self) -> None:
        """显式关闭数据库连接和 HTTP 客户端。"""
        self._conn.close()
        self._llm.http_client.close()

    def total_paid_orders_in_october_test(self) -> None:
        question = "2023年10月总共成交了几笔订单？"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的 Text-to-SQL 助手。"
                    "用户询问订单成交数量时，成交表示订单状态 status = 1（已支付）。"
                    "用户按月份统计订单时，月份字段使用 create_time。"
                    "必须先调用 query_db 查询数据库，再根据查询结果回答。"
                    "只生成只读 SELECT SQL，不要生成 INSERT、UPDATE、DELETE、DROP 等修改语句。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n数据库元数据信息：{self._database_schema}",
            },
        ]
        self._run(messages)

    def _init_database(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(self._database_schema)

        mock_data = [
            (1, 1001, "TSHIRT_1", 50.00, 0, "2023-09-12 10:00:00", None),
            (2, 1001, "TSHIRT_2", 75.50, 1, "2023-09-16 11:00:00", "2023-08-16 12:00:00"),
            (3, 1002, "SHOES_X2", 25.25, 2, "2023-10-17 12:30:00", "2023-08-17 13:00:00"),
            (4, 1003, "SHOES_X2", 25.25, 1, "2023-10-17 12:30:00", "2023-08-17 13:00:00"),
            (5, 1003, "HAT_Z112", 60.75, 1, "2023-10-20 14:00:00", "2023-08-20 15:00:00"),
            (6, 1002, "WATCH_X001", 90.00, 0, "2023-10-28 16:00:00", None),
        ]

        cursor.executemany(
            """
            INSERT INTO orders (id, customer_id, product_id, price, status, create_time, pay_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            mock_data,
        )
        self._conn.commit()

    def _query_db(self, *, query: str) -> list[tuple]:
        sql = query.strip()
        if not sql.lower().startswith("select"):
            raise ValueError(f"只允许执行 SELECT 查询，实际 SQL: {sql}")

        cursor = self._conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def _run(self, messages: list[dict]) -> None:
        # 第一次请求：让模型根据用户问题和表结构决定是否调用 query_db。
        response = self._llm._call_api(messages, tools=self._tools, tool_choice="auto")
        choice = response["choices"][0]["message"]

        if not choice.get("tool_calls"):
            logger.info(f"模型直接回复（未触发工具）: {choice.get('content', '')}")
            return

        # 只把工具调用协议需要的字段放回上下文，避免第一次说明性 content 干扰最终回答。
        messages.append(
            {
                "role": choice["role"],
                "tool_calls": choice["tool_calls"],
            }
        )

        for tool_call in choice["tool_calls"]:
            func_call = tool_call["function"]
            args = json.loads(func_call.get("arguments", "{}"))
            query = args.get("query")
            if not query:
                logger.info(f"工具调用参数不完整，实际参数: {args}")
                return

            logger.info(f"模型生成 SQL: {query}")
            result = self._query_db(query=query)
            logger.info(f"数据库查询结果: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        # 第二次请求：模型根据 SQL 查询结果组织最终业务答案。
        final_response = self._llm._call_api(messages)
        final_answer = final_response["choices"][0]["message"].get("content", "")
        logger.info(f"模型最终回复: {final_answer}")


def main() -> None:
    demo = SqlFunctionCallDemo()
    try:
        demo.total_paid_orders_in_october_test()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
