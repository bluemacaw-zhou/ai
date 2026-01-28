"""
测试线程信息在日志中的显示

演示多线程环境下，每个线程的日志都会显示线程名称和独立的 trace_id
"""

import threading
import time
from ai_starter import get_logger, with_trace

logger = get_logger(__name__)


def worker(worker_id, task_count):
    """工作线程函数"""
    logger.info(f"工作线程 {worker_id} 启动")

    for i in range(task_count):
        logger.info(f"工作线程 {worker_id} 执行任务 {i+1}/{task_count}")
        time.sleep(0.1)

    logger.info(f"工作线程 {worker_id} 完成所有任务")


@with_trace
def process_request(request_id):
    """模拟处理请求（使用装饰器自动分配 trace_id）"""
    logger.info(f"开始处理请求: {request_id}")
    logger.info("验证请求参数")
    time.sleep(0.1)
    logger.info("查询数据库")
    time.sleep(0.1)
    logger.info(f"请求 {request_id} 处理完成")


def main():
    print("=" * 80)
    print("线程日志测试 - 查看每条日志的 [线程名称] 和 [trace_id]")
    print("=" * 80)
    print()

    print("示例1: 主线程日志")
    print("-" * 80)
    logger.info("这是主线程的日志")
    logger.warning("主线程警告信息")
    print()

    print("示例2: 多线程并发执行（每个线程独立 trace_id）")
    print("-" * 80)
    threads = []
    for i in range(3):
        # 为每个线程设置有意义的名称
        t = threading.Thread(target=worker, args=(i, 2), name=f"Worker-{i}")
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    print()
    print("示例3: 使用装饰器处理请求（每次调用新 trace_id）")
    print("-" * 80)
    process_request("REQ-001")
    print()
    process_request("REQ-002")
    print()

    print("=" * 80)
    print("观察要点：")
    print("1. 主线程显示为 [MainThread]")
    print("2. 工作线程显示为 [Worker-0], [Worker-1], [Worker-2]")
    print("3. 每个线程有独立的 trace_id（中括号中的 UUID）")
    print("4. 同一线程内的所有日志共享相同的 trace_id")
    print("5. 使用 @with_trace 装饰器的函数，每次调用都有新的 trace_id")
    print("=" * 80)


if __name__ == "__main__":
    main()
