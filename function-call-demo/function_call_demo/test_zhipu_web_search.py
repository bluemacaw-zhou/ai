"""ZhipuAI 联网搜索演示

演示如何在当前项目框架下使用智谱 Chat Completions 的 web_search 工具：
- 仍然使用 ai-starter 中的 ZhipuAIBase
- 通过 tools 参数开启联网搜索
- 让模型检索最新信息后直接生成答案
"""

from datetime import date

from ai_starter import get_logger
from ai_starter.llm import ZhipuAIBase

logger = get_logger(__name__)


class ZhipuWebSearchDemo:
    """智谱联网搜索问答示例。"""

    _web_search_tools = [
        {
            "type": "web_search",
            "web_search": {
                "enable": True,
                "search_engine": "search_pro",
                "search_result": True,
                "count": 5,
                "search_recency_filter": "noLimit",
                "content_size": "high",
                "search_prompt": (
                    "你正在使用联网搜索结果回答用户问题。"
                    "请优先依据搜索结果 {search_result}，不要只凭模型内置知识回答。"
                    "回答要先给结论，再列出关键亮点，并尽量标注信息来源或发布日期。"
                    f"今天的日期是 {date.today().isoformat()}。"
                ),
            },
        }
    ]

    def __init__(self) -> None:
        self._llm = ZhipuAIBase()

    def close(self) -> None:
        """显式关闭 HTTP 客户端，避免退出时依赖 __del__ 清理资源。"""
        self._llm.http_client.close()

    def latest_question_test(self) -> None:
        messages = [
            {
                "role": "user",
                # "content": "中国央行5月8日执行的5亿元7天逆回购的利率是多少",
                "content": "美国4月非农就业人数增加了多少",
            }
        ]
        self._run(messages)

    def _run(self, messages: list[dict]) -> None:
        # 这里的 web_search 是智谱内置工具，不是我们本地 Python 函数。
        # 模型会在服务端自动检索网页，再把搜索结果融合进最终回答。
        response = self._llm._call_api(
            messages,
            tools=self._web_search_tools,
            tool_choice="auto",
            timeout=120,
        )

        choice = response["choices"][0]["message"]
        answer = choice.get("content", "")
        logger.info(f"模型联网搜索回答: {answer}")


def main() -> None:
    demo = ZhipuWebSearchDemo()
    try:
        demo.latest_question_test()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
