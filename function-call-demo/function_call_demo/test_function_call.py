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
                "description": (
                    "Get current weather by latitude and longitude. "
                    "Use this tool whenever the user asks about weather. "
                    "For well-known cities, infer their latitude and longitude."
                ),
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

    def close(self) -> None:
        """显式关闭 HTTP 客户端，避免退出时依赖 __del__ 清理资源。"""
        self._llm.http_client.close()

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
        messages = [
            {
                "role": "system",
                "content": (
                    "When the user asks about weather, use the get_weather tool. "
                    "For well-known cities, infer their latitude and longitude. "
                    "Do not answer weather questions from general knowledge before using the tool."
                ),
            },
            {"role": "user", "content": "What's the weather like in Shanghai today?"},
        ]
        self._run(messages)

    def no_function_call_test(self) -> None:
        messages = [{"role": "user", "content": "How are you today?"}]
        self._run(messages)

    def missing_args_test(self) -> None:
        messages = [{"role": "user", "content": "What's the weather like today?"}]
        self._run(messages)

    def _run(self, messages: list[dict]) -> None:
        # 第一次请求：把用户问题和工具定义发给模型，让模型判断是否需要调用工具。
        response = self._llm._call_api(messages, tools=self._tools, tool_choice="auto")

        # response 的核心结构大致是：
        # {"choices": [{"message": {"role": "assistant", "content": "...", "tool_calls": [...]}}]}
        # choice 是模型返回的 assistant 消息，可能是普通回答 content，
        # 也可能是工具调用请求 tool_calls。
        choice = response["choices"][0]["message"]

        if choice.get("tool_calls"):
            # 必须把模型返回的 assistant 工具调用消息放回上下文，
            # 但只保留工具调用协议需要的字段，避免把第一次的说明性 content 污染最终回答。
            messages.append(
                {
                    "role": choice["role"],
                    "tool_calls": choice["tool_calls"],
                }
            )
            for tool_call in choice["tool_calls"]:
                # func_call 是模型生成的函数调用信息；arguments 才是入参 JSON 字符串。
                func_call = tool_call["function"]
                args = json.loads(func_call.get("arguments", "{}"))
                required = ["latitude", "longitude"]
                missing = [k for k in required if k not in args]
                if missing:
                    logger.info(f"工具调用参数不完整，缺少: {missing}, 实际参数: {args}")
                    return

                # 执行模型指定的本地函数。模型只决定“调用什么”和“传什么参数”，
                # 真正的函数执行仍然由我们的代码完成。
                logger.info(f"模型触发了工具调用: {func_call}")
                result = self._functions[func_call["name"]](**args)
                logger.info(f"工具执行结果: {result}")

                # 把工具执行结果追加到上下文，并用 tool_call_id 关联到上面的工具调用。
                # content 建议放字符串，这里用 JSON 保留结构化结果，方便模型理解。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            # 第二次请求：模型现在同时看到用户问题、自己的工具调用、工具返回结果，
            # 因此可以基于完整上下文组织一个有头有尾的自然语言答案。
            final_response = self._llm._call_api(messages)
            final_answer = final_response["choices"][0]["message"].get("content", "")
            logger.info(f"模型最终回复: {final_answer}")
        else:
            # 如果模型认为不需要工具，就直接输出第一次请求里的回答。
            answer = choice.get("content", "")
            logger.info(f"模型直接回复（未触发工具）: {answer}")


def main() -> None:
    demo = FunctionCallDemo()
    try:
        demo.function_call_test()
        # demo.no_function_call_test()
        # demo.missing_args_test()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
