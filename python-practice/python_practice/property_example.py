"""
@property 装饰器示例

演示 Python 属性封装的最佳实践：使用 @property 而不是 getter/setter 方法
"""


# ❌ 错误示例 1：直接暴露内部变量（无封装）
class BadPerson:
    """不好的设计：直接暴露变量"""

    def __init__(self, name, age):
        self.name = name
        self.age = age  # 可以被随意修改，无验证


# ❌ 错误示例 2：Java 风格的 getter/setter（不符合 Python 习惯）
class JavaStylePerson:
    """Java 风格：冗长，不 Pythonic"""

    def __init__(self, name, age):
        self._name = name
        self._age = age

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = value

    def get_age(self):
        return self._age

    def set_age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value


# ✅ 正确示例：使用 @property（Python 最佳实践）
class Person:
    """推荐设计：使用 @property 封装属性"""

    def __init__(self, name, age):
        self._name = name  # 私有变量（约定，单下划线）
        self._age = age

    @property
    def name(self):
        """获取姓名（只读属性）"""
        return self._name

    @property
    def age(self):
        """获取年龄"""
        return self._age

    @age.setter
    def age(self, value):
        """设置年龄（带验证）"""
        if value < 0:
            raise ValueError("Age cannot be negative")
        if value > 150:
            raise ValueError("Age too large")
        self._age = value

    @property
    def is_adult(self):
        """计算属性：是否成年（无需存储，每次计算）"""
        return self._age >= 18


# ✅ 进阶示例：延迟初始化（Lazy Loading）
class DataLoader:
    """演示延迟加载：只在第一次访问时加载数据"""

    def __init__(self, file_path):
        self._file_path = file_path
        self._data = None  # 未加载

    @property
    def data(self):
        """延迟加载数据"""
        if self._data is None:
            print(f"Loading data from {self._file_path}...")
            self._data = self._load_data()  # 只加载一次
        return self._data

    def _load_data(self):
        """模拟数据加载（私有方法）"""
        return [1, 2, 3, 4, 5]


# ✅ 实战示例：类似 Text2SQLIndexBuilder 的设计
class QueryBuilder:
    """查询构建器：对外暴露组件，内部私有化"""

    def __init__(self, config):
        self._config = config
        # 私有变量
        self._query_parser = self._build_parser()
        self._executor = self._build_executor()

    @property
    def query_parser(self):
        """获取查询解析器（只读）"""
        return self._query_parser

    @property
    def executor(self):
        """获取执行器（只读）"""
        return self._executor

    def _build_parser(self):
        """构建解析器（私有方法）"""
        return f"Parser with config: {self._config}"

    def _build_executor(self):
        """构建执行器（私有方法）"""
        return f"Executor with config: {self._config}"


def demo_basic_usage():
    """基础用法演示"""
    print("=" * 60)
    print("基础用法")
    print("=" * 60)

    person = Person("张三", 25)

    # ✅ 访问属性（像普通属性一样）
    print(f"姓名: {person.name}")  # 调用 @property getter
    print(f"年龄: {person.age}")
    print(f"是否成年: {person.is_adult}")

    # ✅ 修改属性（通过 setter，自动验证）
    person.age = 30
    print(f"修改后年龄: {person.age}")

    # ❌ 尝试设置非法值
    try:
        person.age = -5
    except ValueError as e:
        print(f"验证失败: {e}")

    # ❌ 尝试修改只读属性
    try:
        person.name = "李四"  # 没有 setter，会报错
    except AttributeError as e:
        print(f"无法修改只读属性: {e}")

    print()


def demo_lazy_loading():
    """延迟加载演示"""
    print("=" * 60)
    print("延迟加载")
    print("=" * 60)

    loader = DataLoader("data.txt")

    print("1. 创建 DataLoader（数据未加载）")

    print("2. 第一次访问 data 属性（触发加载）")
    data1 = loader.data

    print("3. 第二次访问 data 属性（直接返回缓存）")
    data2 = loader.data

    print(f"数据: {data1}")
    print(f"两次访问返回同一对象: {data1 is data2}")

    print()


def demo_query_builder():
    """查询构建器演示"""
    print("=" * 60)
    print("查询构建器（类似 Text2SQLIndexBuilder）")
    print("=" * 60)

    builder = QueryBuilder(config="prod")

    # ✅ 访问组件（通过 @property）
    parser = builder.query_parser
    executor = builder.executor

    print(f"Parser: {parser}")
    print(f"Executor: {executor}")

    # ❌ 无法修改（只读）
    try:
        builder.query_parser = "new parser"
    except AttributeError as e:
        print(f"无法修改只读属性: {e}")

    print()


def compare_styles():
    """对比不同风格"""
    print("=" * 60)
    print("对比不同风格")
    print("=" * 60)

    # Java 风格（冗长）
    java_person = JavaStylePerson("张三", 25)
    print(f"Java 风格: {java_person.get_name()}, {java_person.get_age()}")
    java_person.set_age(30)

    # Python 风格（简洁）
    py_person = Person("李四", 25)
    print(f"Python 风格: {py_person.name}, {py_person.age}")
    py_person.age = 30

    print("\n✅ Python @property 风格更简洁、更易读！")
    print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_lazy_loading()
    demo_query_builder()
    compare_styles()

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("✅ 使用 @property 的优势：")
    print("  1. 封装：变量私有化（_variable）")
    print("  2. 简洁：访问时像普通属性（obj.attr）")
    print("  3. 验证：在 setter 中添加验证逻辑")
    print("  4. 灵活：可以从属性升级到 @property 而不改使用代码")
    print("  5. 计算：可以实现计算属性（每次访问时重新计算）")
    print()
    print("❌ 避免：")
    print("  1. 不要用 Java 风格的 get_xxx() / set_xxx()")
    print("  2. 不要直接暴露重要的内部变量")
