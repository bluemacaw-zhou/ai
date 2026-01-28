from fastmcp import FastMCP
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Demo 🚀")

@mcp.tool()
def add(a: int, b: int) -> int:
    """执行两个整数的加法运算

    这个工具接收两个整数参数并返回它们的和。
    适用场景：
    - 需要计算两个数字的总和
    - 数学运算和数值计算
    - 累加计数等场景

    Args:
        a: 第一个加数（整数）
        b: 第二个加数（整数）

    Returns:
        int: 两个数的和 (a + b)

    Examples:
        add(2, 3) -> 5
        add(-1, 1) -> 0
        add(100, 200) -> 300
    """
    # 记录请求参数
    logger.info(f"调用add工具 - 参数: a={a}, b={b}")

    result = a + b
    logger.info(f"add工具执行结果: {a} + {b} = {result}")

    return result

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """健康检查端点 - 用于容器健康检查和监控"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "service": "fastmcp_demo",
        "version": "1.0.0",
        "transport": "http"
    })

if __name__ == "__main__":
    mcp.run()
