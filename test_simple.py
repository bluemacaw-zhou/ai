import chromadb
import os
from chromadb.config import Settings

print("开始测试...")

os.environ['NO_PROXY'] = '192.168.254.129'
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

print("环境变量设置完成")

try:
    print("正在创建客户端...")
    client = chromadb.HttpClient(
        host="192.168.254.129",
        port=18000,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
            chroma_client_auth_credentials="admin:admin"
        )
    )
    print("客户端创建成功")
    
    print("正在测试连接...")
    result = client.heartbeat()
    print(f"连接测试结果: {result}")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("测试结束")
