"""
测试 ai-starter 共享包

演示如何导入和使用 say_hello 方法
"""

from ai_starter import say_hello


def main():
    """主函数"""
    print("=" * 50)
    print("测试 ai-starter 共享包")
    print("=" * 50)

    # 调用 say_hello 方法
    result = say_hello()

    print(f"\n调用 say_hello() 返回: {result}")
    print("\n测试完成！")


if __name__ == "__main__":
    main()
