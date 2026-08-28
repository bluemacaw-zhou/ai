"""MCP 工具来源配置 (wind mcp server)。

从 ag-ui-python-backend 的 config/mcp_settings.py 移植并改造：
源项目配置根为 ``ag_ui_backend.mcp.servers``；本项目采用扁平结构，
配置根改为 ``mcp.servers``（与 ``llm.agents`` 平级）。

对应 config.yaml (mcp.servers 是一个 map，key 为 server 名):
    mcp.servers.<name>.url             wind server 地址
    mcp.servers.<name>.session_id       客户端未传 sessionId 时的默认值
    mcp.servers.<name>.session_headers  sessionId 注入到哪些 header
    mcp.servers.<name>.http             该 server 的 http (代理/超时/SSL)

"配了就加载": servers map 里列出的全部加载; 注释掉或删除即不加载 (不另设开关)。
本地 @tool 工具不在这里配，由 mcp_tools/local 目录扫描自动加载。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .app_config import Config

# 配置根路径。改这里即可整体迁移配置节点。
_SERVERS_ROOT = "mcp.servers"


@dataclass(frozen=True)
class McpHttpSettings:
    """单个 server 的 http 段，直接映射 (代理启不启用、选哪个由消费方判断)。"""

    proxy_enabled: bool = False
    proxy_http: str | None = None
    proxy_https: str | None = None
    timeout: float = 60.0
    verify_ssl: bool = True


@dataclass(frozen=True)
class McpServerSettings:
    """单个 wind mcp server 的连接配置，直接映射配置 (name 即 map 的 key)。"""

    name: str
    url: str = ""
    session_id: str | None = None
    session_headers: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    adapter: str = "wind"
    http: McpHttpSettings = field(default_factory=McpHttpSettings)


@dataclass(frozen=True)
class McpSettings:
    """MCP 工具来源配置 (servers 是 name -> server 的 map)。"""

    servers: dict[str, McpServerSettings]

    @classmethod
    def load(cls) -> "McpSettings":
        raw_servers = Config().get(_SERVERS_ROOT, {}) or {}
        servers = {
            name: _server_settings_from_config_item(name, item)
            for name, item in raw_servers.items()
        }
        return cls(servers=servers)


def _server_settings_from_config_item(name: str, item: dict) -> McpServerSettings:
    http = item.get("http", {}) or {}
    proxy = http.get("proxy", {}) or {}
    return McpServerSettings(
        name=name,
        url=item.get("url", ""),
        session_id=item.get("session_id") or item.get("sessionId"),
        session_headers=tuple(item.get("session_headers", []) or []),
        tool_names=tuple(item.get("tool_names", []) or []),
        adapter=str(item.get("adapter", "wind") or "wind"),
        http=McpHttpSettings(
            proxy_enabled=bool(proxy.get("enabled", False)),
            proxy_http=proxy.get("http"),
            proxy_https=proxy.get("https"),
            timeout=float(http.get("timeout", 60)),
            verify_ssl=bool(http.get("verify_ssl", True)),
        ),
    )
