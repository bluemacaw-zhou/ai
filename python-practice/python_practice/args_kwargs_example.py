"""
函数参数定义与解包示例

演示 Python 中 * 和 ** 在参数定义和调用两个场景下的含义：
- 定义时：收集多余参数
- 调用时：展开容器为参数
"""


# ==================== 维度一：是否必传 ====================

def required_args(arg1, arg2, *, arg3, arg4):
    """四个参数均必传（无默认值）"""
    return f"arg1={arg1}, arg2={arg2}, arg3={arg3}, arg4={arg4}"


def optional_args(arg1, arg2=0, *, arg3, arg4=0):
    """arg2 和 arg4 有默认值，可不传"""
    return f"arg1={arg1}, arg2={arg2}, arg3={arg3}, arg4={arg4}"


# ==================== 维度二：位置 vs 名称 ====================

def positional_and_keyword(arg1, arg2, *, arg3, arg4):
    """
    * 左边：可以按位置传，也可以按名称传
    * 右边：只能按名称传，禁止按位置传
    """
    return f"arg1={arg1}, arg2={arg2}, arg3={arg3}, arg4={arg4}"


# ==================== 定义时的 * 和 ** ====================

def collect_args(*args, **kwargs):
    """
    *args  : 把多余的位置参数收集成元组
    **kwargs: 把多余的关键字参数收集成字典
    """
    return f"args={args}, kwargs={kwargs}"


# ==================== 调用时的 * 和 ** ====================

def demo_unpack() -> None:
    """演示调用时用 * 和 ** 展开容器"""

    args = (1, 2)               # 元组 → 展开为位置参数
    kwargs = {"arg3": 3, "arg4": 4}  # 字典 → 展开为关键字参数

    # *args 展开 → arg1=1, arg2=2
    # **kwargs 展开 → arg3=3, arg4=4
    result = required_args(*args, **kwargs)
    print(result)

    # 等价于
    result2 = required_args(1, 2, arg3=3, arg4=4)
    print(result2)


# ==================== 综合对比 ====================

def demo_all() -> None:

    # ✓ 正常：位置 + 关键字
    print(positional_and_keyword(1, 2, arg3=3, arg4=4))

    # ✓ 正常：全用名称
    print(positional_and_keyword(arg1=1, arg2=2, arg3=3, arg4=4))

    # ✓ 正常：解包传入
    args = (1, 2)
    kwargs = {"arg3": 3, "arg4": 4}
    print(positional_and_keyword(*args, **kwargs))

    # ✓ 收集参数
    print(collect_args(1, 2, arg3=3, arg4=4))
    # 输出：args=(1, 2), kwargs={'arg3': 3, 'arg4': 4}

    # ✗ 错误：* 右边不能按位置传（取消注释会报错）
    # positional_and_keyword(1, 2, 3, 4)

    # ✗ 错误：必传参数缺少（取消注释会报错）
    # positional_and_keyword(1, 2, arg4=4)


if __name__ == "__main__":
    print("=" * 50)
    print("解包演示")
    print("=" * 50)
    demo_unpack()

    print()
    print("=" * 50)
    print("综合对比")
    print("=" * 50)
    demo_all()
