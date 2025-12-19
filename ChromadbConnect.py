import chromadb
import os
from chromadb.config import Settings
import numpy as np


os.environ['NO_PROXY'] = '192.168.254.129'
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# 硬编码的嵌入向量（384维，模拟 all-MiniLM-L6-v2 的输出）
# 为每个雨具句子预设的向量表示
rain_gear_embeddings = {
    "雨伞是最常见的雨具，可以遮挡雨水": [0.1, -0.2, 0.3, 0.1, -0.1, 0.4, -0.3, 0.2, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 0.2, -0.1, 0.4, -0.3, 0.1, 0.2, -0.2, 0.3, 0.1, -0.1, 0.4, -0.3, 0.2, 0.1] + [0.0] * 352,
    "雨衣能够保护全身不被雨水淋湿": [0.2, -0.1, 0.4, 0.2, -0.2, 0.5, -0.2, 0.3, 0.2, -0.1, 0.4, -0.2, 0.3, -0.2, 0.5, 0.2, -0.1, 0.4, 0.3, -0.2, 0.5, -0.2, 0.2, 0.3, -0.1, 0.4, 0.2, -0.2, 0.5, -0.2, 0.3, 0.2] + [0.1] * 352,
    "雨靴防水性能好，适合在雨天穿着": [0.3, -0.3, 0.2, 0.3, -0.1, 0.3, -0.4, 0.1, 0.3, -0.3, 0.2, -0.1, 0.1, -0.4, 0.3, 0.3, -0.3, 0.2, 0.1, -0.1, 0.3, -0.4, 0.3, 0.1, -0.3, 0.2, 0.3, -0.1, 0.3, -0.4, 0.1, 0.3] + [0.2] * 352,
    "雨帽可以保护头部免受雨水": [0.1, -0.4, 0.1, 0.1, -0.3, 0.2, -0.1, 0.4, 0.1, -0.4, 0.1, -0.3, 0.4, -0.1, 0.2, 0.1, -0.4, 0.1, 0.4, -0.3, 0.2, -0.1, 0.1, 0.4, -0.4, 0.1, 0.1, -0.3, 0.2, -0.1, 0.4, 0.1] + [-0.1] * 352,
    "防水包能保护包内物品不受潮": [0.4, -0.1, 0.3, 0.4, -0.2, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, -0.2, 0.2, -0.3, 0.1, 0.4, -0.1, 0.3, 0.2, -0.2, 0.1, -0.3, 0.4, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, -0.3, 0.2, 0.4] + [0.3] * 352
}

client = chromadb.HttpClient(
    host="192.168.254.129",
    port=18000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials="admin:admin"  # 改成你的密码
        # chroma_client_auth_provider=None,
        # chroma_client_auth_credentials=None  # 改成你的密码
    )
)

# 测试连接
print("🔄 正在测试连接...")
try:
    heartbeat = client.heartbeat()
    print(f"✅ 连接成功: {heartbeat}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# 创建或获取集合
print("🔄 正在创建或获取集合...")
try:
    collection = client.get_or_create_collection("rain_gear_collection")
    print("✅ 集合创建/获取成功")
except Exception as e:
    print(f"❌ 集合创建失败: {e}")
    exit(1)

# 准备文档和对应的嵌入向量
# documents = [
#     "雨伞是最常见的雨具，可以遮挡雨水",
#     "雨衣能够保护全身不被雨水淋湿",
#     "雨靴防水性能好，适合在雨天穿着",
#     "雨帽可以保护头部免受雨水",
#     "防水包能保护包内物品不受潮"
# ]

# print("🔄 正在准备嵌入向量...")
# embeddings = [rain_gear_embeddings[doc] for doc in documents]
# print(f"✅ 准备了 {len(embeddings)} 个向量，每个向量维度: {len(embeddings[0])}")

# 插入雨具数据（使用硬编码的向量）
# print("🔄 正在插入雨具数据...")
# try:
#     collection.add(
#         documents=documents,
#         embeddings=embeddings,
#         metadatas=[
#             {"type": "umbrella", "waterproof": True, "portable": True},
#             {"type": "raincoat", "waterproof": True, "portable": False},
#             {"type": "rain_boots", "waterproof": True, "portable": False},
#             {"type": "rain_hat", "waterproof": True, "portable": True},
#             {"type": "waterproof_bag", "waterproof": True, "portable": True}
#         ],
#         ids=["umbrella_001", "raincoat_001", "boots_001", "hat_001", "bag_001"]
#     )
#     print("✅ 雨具数据插入成功！")
# except Exception as e:
#     print(f"❌ 数据插入失败: {e}")
#     exit(1)

# 验证插入结果
print("🔄 正在进行向量查询...")
try:
    # 使用"雨伞"对应的向量进行查询
    query_embedding = rain_gear_embeddings["雨伞是最常见的雨具，可以遮挡雨水"]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    print(f"✅ 向量查询结果: {results}")
except Exception as e:
    print(f"❌ 向量查询失败: {e}")

print("🔄 正在获取所有文档...")
try:
    # 也可以直接按文档内容查询
    results_by_text = collection.get()
    print(f"✅ 所有文档: {results_by_text}")
except Exception as e:
    print(f"❌ 获取文档失败: {e}")

print("🎉 测试完成！")
