"""通用配置读取类（类似 Spring Boot 的配置加载）。

从 ag-ui-python-backend 的 common/config.py 移植：
    - 单例，全局唯一
    - 懒加载：第一次访问配置时自动加载
    - 支持从项目根目录（含 pyproject.toml）查找 config.yaml/yml/json/toml
    - 支持点号分隔的嵌套键取值
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """通用配置读取类。

    支持从 YAML、JSON、TOML 文件读取配置；单例模式；懒加载。
    """

    _instance: Optional["Config"] = None
    _config_data: Dict[str, Any] = {}
    _config_file: Optional[str] = None
    _loaded: bool = False

    # 配置文件查找顺序
    _config_names = ["config.yaml", "config.yml", "config.json", "config.toml"]

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls) -> "Config":
        """手动加载配置文件（通常无需调用，首次访问会自动懒加载）。

        查找逻辑：
        1. 从当前目录向上查找 pyproject.toml 所在目录（项目根目录）
        2. 从项目根目录按顺序查找 config.yaml/yml/json/toml

        Raises:
            FileNotFoundError: 找不到 pyproject.toml 或配置文件
        """
        config = cls()

        if config._loaded:
            return config

        project_root = cls._find_project_root()
        if project_root is None:
            raise FileNotFoundError(
                "找不到项目根目录。请确保在包含 pyproject.toml 的项目目录中运行。\n"
                f"当前工作目录: {os.getcwd()}"
            )

        config_file = cls._find_config_file()
        if config_file is None:
            raise FileNotFoundError(
                "配置文件未找到。请在项目根目录创建以下任一文件：\n"
                "  - config.yaml\n"
                "  - config.yml\n"
                "  - config.json\n"
                "  - config.toml\n"
                f"项目根目录: {project_root}\n"
                f"当前工作目录: {os.getcwd()}"
            )

        return config._load_from_file(config_file)

    def _ensure_loaded(self) -> None:
        """确保配置已加载（懒加载）。"""
        if not self._loaded:
            self.load()

    def load_file(self, config_file: str) -> "Config":
        """从指定路径加载配置文件（用于测试或特殊场景）。"""
        return self._load_from_file(config_file)

    def _load_from_file(self, config_file: str) -> "Config":
        """从文件加载配置的内部方法。"""
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        self._config_file = str(config_path)
        suffix = config_path.suffix.lower()

        if suffix == ".json":
            self._load_json(config_path)
        elif suffix in [".yaml", ".yml"]:
            self._load_yaml(config_path)
        elif suffix == ".toml":
            self._load_toml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}")

        self._loaded = True
        return self

    @classmethod
    def _find_project_root(cls) -> Optional[Path]:
        """向上递归查找包含 pyproject.toml 的目录作为项目根目录。"""
        current = Path.cwd().resolve()
        for directory in [current] + list(current.parents):
            if (directory / "pyproject.toml").exists():
                return directory
        return None

    @classmethod
    def _find_config_file(cls) -> Optional[str]:
        """从项目根目录查找配置文件。"""
        project_root = cls._find_project_root()
        if not project_root:
            return None

        for config_name in cls._config_names:
            config_path = project_root / config_name
            if config_path.exists() and config_path.is_file():
                return str(config_path)

        return None

    def _load_json(self, path: Path) -> None:
        """加载 JSON 配置文件。"""
        with open(path, "r", encoding="utf-8") as f:
            self._config_data = json.load(f)

    def _load_yaml(self, path: Path) -> None:
        """加载 YAML 配置文件。"""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "需要安装 PyYAML 来读取 YAML 配置文件\n请运行: uv add pyyaml"
            ) from exc
        with open(path, "r", encoding="utf-8") as f:
            self._config_data = yaml.safe_load(f) or {}

    def _load_toml(self, path: Path) -> None:
        """加载 TOML 配置文件。"""
        try:
            import tomllib as toml_reader  # Python 3.11+
        except ImportError:
            try:
                import tomli as toml_reader  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "需要 Python 3.11+ 的 tomllib 或安装 tomli 来读取 TOML 配置文件\n"
                    "请运行: uv add tomli"
                ) from exc
        with open(path, "rb") as f:
            self._config_data = toml_reader.load(f)

    def load_from_dict(self, config_dict: Dict[str, Any]) -> "Config":
        """从字典加载配置（用于测试或动态配置）。"""
        self._config_data = config_dict
        self._loaded = True
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项（支持点号分隔的嵌套键，懒加载）。

        Examples:
            >>> Config().get("llm.defaults.api_base")
            >>> Config().get("server.port", 8000)
            >>> Config().get("server.cors_allow_origins", [])
        """
        self._ensure_loaded()

        keys = key.split(".")
        value: Any = self._config_data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_list(self, key: str, default: list | None = None) -> list:
        """获取列表类型的配置项（支持点号分隔的嵌套键）。

        Examples:
            >>> Config().get_list("server.cors_allow_origins", [])
        """
        value = self.get(key, default)
        if value is None:
            return [] if default is None else default
        if isinstance(value, list):
            return value
        # 如果是字符串（逗号分隔），则拆分
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [] if default is None else list(default)

    def get_required(self, key: str) -> Any:
        """获取必需的配置项（不存在则抛出 KeyError）。"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"必需的配置项不存在: {key}")
        return value

    def has(self, key: str) -> bool:
        """检查配置项是否存在。"""
        return self.get(key) is not None

    def all(self) -> Dict[str, Any]:
        """获取所有配置（副本）。"""
        self._ensure_loaded()
        return self._config_data.copy()

    def reload(self) -> "Config":
        """重新加载配置文件。"""
        if not self._config_file:
            raise RuntimeError("没有已加载的配置文件")
        self._loaded = False
        return self._load_from_file(self._config_file)

    @classmethod
    def reset(cls) -> None:
        """重置单例（主要用于测试）。"""
        cls._instance = None
        cls._config_data = {}
        cls._config_file = None
        cls._loaded = False

    @classmethod
    def is_loaded(cls) -> bool:
        """检查配置是否已加载。"""
        return cls._loaded


def load_config(config_file: str | None = None) -> Config:
    """加载配置文件到全局单例（便捷函数）。

    如果不指定 config_file，则自动查找配置文件。
    """
    if config_file is None:
        return Config.load()
    return Config().load_file(config_file)


def find_config_file() -> Optional[str]:
    """从项目根目录查找配置文件（便捷函数）。"""
    return Config._find_config_file()
