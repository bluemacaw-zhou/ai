"""Function Calling 演示

演示如何用 ZhipuAI GLM 实现 Function Calling（工具调用），
与 OpenAI 的接口格式完全一致。
"""

import json

from ai_starter import get_logger
from ai_starter.llm import ZhipuAIBase

logger = get_logger(__name__)


class FunctionCallDemo:

    _tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for provided coordinates in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": False,
                },
                # "strict": True,  # OpenAI 专有字段，ZhipuAI 不支持
            },
        }
    ]

    _functions = {
        "get_weather": lambda **kwargs: FunctionCallDemo._get_weather(**kwargs),
    }

    def __init__(self) -> None:
        self._llm = ZhipuAIBase()

    @staticmethod
    def _get_weather(*, latitude: float, longitude: float) -> dict:
        """模拟天气查询（实际项目中调用真实天气 API）"""
        return {
            "temperature": 23,
            "weather": "Sunny",
            "wind_direction": "South",
            "windy": 2,
        }

    def function_call_test(self) -> None:
        messages = [{"role": "user", "content": "What's the weather like in Shanghai today?"}]
        self._run(messages)

    def no_function_call_test(self) -> None:
        messages = [{"role": "user", "content": "How are you today?"}]
        self._run(messages)

    def missing_args_test(self) -> None:
        messages = [{"role": "user", "content": "What's the weather like today?"}]
        self._run(messages)

    def _run(self, messages: list[dict]) -> None:
        response = self._llm._call_api(messages, tools=self._tools, tool_choice="auto")
        choice = response["choices"][0]["message"]
        if choice.get("tool_calls"):
            func_call = choice["tool_calls"][0]["function"]
            args = json.loads(func_call.get("arguments", "{}"))
            required = ["latitude", "longitude"]
            missing = [k for k in required if k not in args]
            if missing:
                logger.info(f"工具调用参数不完整，缺少: {missing}, 实际参数: {args}")
            else:
                logger.info(f"模型触发了工具调用: {func_call}")
                result = self._functions[func_call["name"]](**args)
                logger.info(f"工具执行结果: {result}")
        else:
            logger.info(f"模型直接回复（未触发工具）: {choice.get('content', '')}")


def main() -> None:
    demo = FunctionCallDemo()
    demo.function_call_test()
    # demo.no_function_call_test()
    # demo.missing_args_test()


if __name__ == "__main__":
    main()
