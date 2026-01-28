"""
日志使用示例

演示如何使用ai-starter的日志功能（基于Python标准logging模块）

注意：
- 链路追踪功能默认开启
- 每个线程自动获得独立的 trace_id
- 日志格式已固定，不可配置
- 日志级别可通过参数或配置文件设置
"""

import threading
import time
from ai_starter import get_logger, setup_logging_from_config, trace_context, with_trace


def example_basic_logging():
    """基础日志使用"""
    print("=== 示例1: 基础日志使用 ===\n")

    # 创建logger
    logger = get_logger(__name__)

    # 不同级别的日志
    logger.debug("这是DEBUG级别日志 - 调试详细信息")
    logger.info("这是INFO级别日志 - 一般信息")
    logger.warning("这是WARNING级别日志 - 警告信息")
    logger.error("这是ERROR级别日志 - 错误信息")
    logger.critical("这是CRITICAL级别日志 - 严重错误")

    print("\n" + "=" * 60 + "\n")


def example_with_level():
    """指定日志级别"""
    print("=== 示例2: 指定日志级别 ===\n")

    # 创建DEBUG级别的logger
    logger = get_logger(__name__ + ".debug", level="DEBUG")

    logger.debug("DEBUG级别可见")
    logger.info("INFO级别可见")
    logger.warning("WARNING级别可见")

    print("\n" + "=" * 60 + "\n")


def example_with_file():
    """日志输出到文件"""
    print("=== 示例3: 日志输出到文件 ===\n")

    # 创建输出到文件的logger
    logger = get_logger(
        __name__ + ".file",
        level="INFO",
        log_file="logs/app.log"
    )

    logger.info("这条日志会同时输出到控制台和文件")
    logger.warning("日志文件: logs/app.log")

    print("✅ 日志已写入 logs/app.log\n")
    print("=" * 60 + "\n")


def example_auto_trace():
    """自动链路追踪示例"""
    print("=== 示例4: 自动链路追踪 ===\n")

    logger = get_logger(__name__ + ".auto_trace")

    # 不需要手动设置，每个线程自动获得 trace_id
    logger.info("第一条日志 - 自动生成 trace_id")
    logger.info("第二条日志 - 使用相同的 trace_id")
    logger.info("第三条日志 - 同一线程共享 trace_id")

    print("注意: 上面三条日志的 trace_id 是相同的\n")
    print("=" * 60 + "\n")


def example_multi_thread():
    """多线程链路追踪示例"""
    print("=== 示例5: 多线程独立 trace_id ===\n")

    logger = get_logger(__name__ + ".multi_thread")

    def worker(worker_id):
        logger.info(f"工作线程 {worker_id} 开始")
        time.sleep(0.1)
        logger.info(f"工作线程 {worker_id} 处理中")
        time.sleep(0.1)
        logger.info(f"工作线程 {worker_id} 完成")

    # 创建多个线程
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    print("注意: 每个线程都有独立的 trace_id\n")
    print("=" * 60 + "\n")


def example_with_decorator():
    """使用装饰器的链路追踪示例"""
    print("=== 示例6: @with_trace 装饰器 ===\n")

    logger = get_logger(__name__ + ".decorator")

    @with_trace
    def handle_request(user_id):
        """模拟处理 HTTP 请求"""
        logger.info(f"开始处理用户请求: {user_id}")
        logger.info("验证用户权限")
        logger.info("查询数据库")
        logger.info(f"请求处理完成: {user_id}")
        return f"Success for {user_id}"

    # 每次调用都会获得新的 trace_id
    handle_request(123)
    print()
    handle_request(456)
    print()
    handle_request(789)

    print("注意: 每次函数调用都有独立的 trace_id\n")
    print("=" * 60 + "\n")


def example_trace_context():
    """手动控制链路追踪示例"""
    print("=== 示例7: 手动控制 trace_id（高级用法）===\n")

    logger = get_logger(__name__ + ".manual_trace")

    # 场景：从 HTTP 请求头传递 trace_id
    incoming_trace_id = "req-from-frontend-abc123"

    with trace_context(incoming_trace_id) as trace_id:
        logger.info(f"使用传入的 trace_id: {trace_id}")
        logger.info("处理来自前端的请求")

    print("\n" + "=" * 60 + "\n")


def example_from_config():
    """从配置文件读取日志级别"""
    print("=== 示例8: 从配置文件读取日志级别 ===\n")

    try:
        # logger会自动从配置文件读取 logging.level 和 logging.file
        # 不需要手动传入config参数
        logger = get_logger(__name__ + ".config")
        logger.info("日志级别从配置文件自动读取")
        logger.debug("如果配置是INFO，这条DEBUG日志不会显示")

        print("提示: 在 config.yaml 中配置:")
        print("  logging:")
        print("    level: DEBUG")
        print("    file: logs/app.log")

    except FileNotFoundError:
        print("⚠️  配置文件不存在")
        print("可以复制 config.example.yaml 为 config.yaml")

    print("\n" + "=" * 60 + "\n")


def example_multiple_loggers():
    """多个logger的使用"""
    print("=== 示例9: 多个logger ===\n")

    # 不同模块使用不同logger
    logger_db = get_logger("myapp.database", level="DEBUG")
    logger_api = get_logger("myapp.api", level="INFO")

    logger_db.debug("数据库连接调试信息")
    logger_db.info("数据库查询成功")

    logger_api.info("API请求处理")
    logger_api.warning("API响应缓慢")

    print("\n" + "=" * 60 + "\n")


def example_exception_logging():
    """异常日志记录"""
    print("=== 示例10: 异常日志记录 ===\n")

    logger = get_logger(__name__ + ".exception")

    try:
        # 模拟一个错误
        result = 10 / 0
    except Exception as e:
        # 记录异常堆栈
        logger.exception("发生了一个除零错误")
        # 或者
        logger.error(f"错误信息: {e}", exc_info=True)

    print("\n" + "=" * 60 + "\n")


def example_structured_logging():
    """结构化日志（使用extra参数）"""
    print("=== 示例11: 结构化日志 ===\n")

    logger = get_logger(__name__ + ".structured")

    # 使用extra传递额外信息
    logger.info(
        "用户登录",
        extra={
            "user_id": 12345,
            "ip": "192.168.1.100",
            "action": "login"
        }
    )

    logger.warning(
        "API调用超时",
        extra={
            "endpoint": "/api/users",
            "timeout": 30,
            "retry_count": 3
        }
    )

    print("\n" + "=" * 60 + "\n")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("日志使用示例 - 基于Python标准logging模块")
    print("=" * 60)
    print()

    example_basic_logging()
    example_with_level()
    example_with_file()
    example_auto_trace()
    example_multi_thread()
    example_with_decorator()
    example_trace_context()
    example_from_config()
    example_multiple_loggers()
    example_exception_logging()
    example_structured_logging()

    print("=" * 60)
    print("✅ 所有示例运行完成")
    print("=" * 60)
    print()
    print("日志格式（固定）:")
    print("  %(asctime)s - [%(trace_id)s] - %(levelname)-5s - [%(threadName)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s")
    print()
    print("格式说明:")
    print("  时间戳 - [trace_id] - 级别 - [线程名] - 文件:函数:行号 - 消息")
    print()
    print("日志级别说明:")
    print("  DEBUG    - 详细的调试信息")
    print("  INFO     - 一般信息消息（默认）")
    print("  WARNING  - 警告信息")
    print("  ERROR    - 错误信息")
    print("  CRITICAL - 严重错误")
    print()
    print("常用方法:")
    print("  logger.debug(msg)     - 调试日志")
    print("  logger.info(msg)      - 信息日志")
    print("  logger.warning(msg)   - 警告日志")
    print("  logger.error(msg)     - 错误日志")
    print("  logger.critical(msg)  - 严重错误日志")
    print("  logger.exception(msg) - 记录异常堆栈")
    print()
    print("链路追踪:")
    print("  - 每个线程自动获得独立的 trace_id")
    print("  - 使用 @with_trace 装饰器为函数分配新 trace_id")
    print("  - 使用 trace_context() 手动控制 trace_id（高级用法）")
    print()
    print("适用场景:")
    print("  - HTTP 请求处理：每个请求独立 trace_id")
    print("  - MQ 消息处理：每个消息独立 trace_id")
    print("  - 多线程任务：每个线程独立 trace_id")


if __name__ == "__main__":
    main()
