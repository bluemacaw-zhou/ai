"""
Config使用示例

演示如何使用Config类加载配置并初始化各种组件
"""

from ai_starter import (
    Config,
    ChromaDB,
    GLMEmbedding
)


def example_auto_load():
    """自动加载配置（类似 Spring Boot）"""
    print("=== 示例1: 自动加载配置 ===\n")

    # 方式1: 类方法自动加载（推荐）
    Config.load()  # 自动查找并加载 config.yaml

    # 读取配置
    api_key = Config().get("api.zhipuai.key")
    print(f"ZhipuAI API Key: {api_key[:20] if api_key else 'Not configured'}...")

    llm_model = Config().get("models.llm.model", "glm-4-flash")
    print(f"LLM Model: {llm_model}")

    print()


def example_components():
    """组件自动读取配置"""
    print("=== 示例2: 组件自动读取配置 ===\n")

    # 直接创建，组件内部会自动加载配置（懒加载）
    # 不需要手动调用 Config.load()
    db = ChromaDB()
    embedding = GLMEmbedding()

    print(f"ChromaDB: {db.host}:{db.port}")
    print(f"Embedding: {embedding.get_model_name()}")

    print()


def example_override_config():
    """覆盖配置文件中的值"""
    print("=== 示例3: 覆盖配置 ===\n")

    # 覆盖配置文件中的 model 参数
    # 参数优先级：函数参数 > 配置文件 > 默认值
    embedding = GLMEmbedding(model="embedding-3")

    print(f"使用覆盖后的模型: {embedding.get_model_name()}")

    print()


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Config 使用示例（Spring Boot 风格）")
    print("=" * 60)
    print()

    try:
        example_auto_load()
        example_components()
        example_override_config()

        print("=" * 60)
        print("✅ 所有示例运行完成")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
