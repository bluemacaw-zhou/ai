import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    通用配置读取类（类似 Spring Boot 的配置加载）

    支持从YAML、JSON、TOML文件读取配置
    单例模式，确保配置在全局唯一
    懒加载：第一次访问配置时自动加载
    """

    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    _config_file: Optional[str] = None
    _loaded: bool = False

    # 配置文件查找顺序
    _config_names = ["config.yaml", "config.yml", "config.json", "config.toml"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls) -> 'Config':
        """
        手动加载配置文件

        从当前工作目录按顺序查找配置文件：
        1. config.yaml
        2. config.yml
        3. config.json
        4. config.toml

        Returns:
            Config实例（支持链式调用）

        Raises:
            FileNotFoundError: 当前目录下找不到任何配置文件

        Note:
            通常不需要手动调用此方法，配置会在第一次访问时自动加载。
            此方法用于需要显式控制加载时机或重新加载配置的场景。

        Examples:
            >>> # 显式加载（可选）
            >>> Config.load()
            >>> # 之后访问配置
            >>> Config().get("api.key")
        """
        config = cls()

        if config._loaded:
            # logger.debug("配置已加载，跳过重复加载")
            return config

        config_file = cls._find_config_file()
        if config_file is None:
            raise FileNotFoundError(
                "配置文件未找到。请确保当前目录下有以下任一文件：\n"
                "  - config.yaml\n"
                "  - config.yml\n"
                "  - config.json\n"
                "  - config.toml\n"
                f"当前工作目录: {os.getcwd()}"
            )

        return config._load_from_file(config_file)

    def _ensure_loaded(self):
        """确保配置已加载（懒加载）"""
        if not self._loaded:
            self.load()

    def load_file(self, config_file: str) -> 'Config':
        """
        从指定路径加载配置文件（用于测试或特殊场景）

        Args:
            config_file: 配置文件路径

        Returns:
            Config实例（支持链式调用）

        Examples:
            >>> Config().load_file("/path/to/custom-config.yaml")
        """
        return self._load_from_file(config_file)

    def _load_from_file(self, config_file: str) -> 'Config':
        """从文件加载配置的内部方法"""
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        self._config_file = str(config_path)
        suffix = config_path.suffix.lower()

        if suffix == '.json':
            self._load_json(config_path)
        elif suffix in ['.yaml', '.yml']:
            self._load_yaml(config_path)
        elif suffix == '.toml':
            self._load_toml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}")

        self._loaded = True
        # 配置加载成功，静默完成（避免循环依赖）
        return self

    @classmethod
    def _find_config_file(cls) -> Optional[str]:
        """
        从当前工作目录查找配置文件

        Returns:
            找到的配置文件路径，未找到返回 None
        """
        current_dir = Path(os.getcwd()).resolve()

        for config_name in cls._config_names:
            config_path = current_dir / config_name
            if config_path.exists() and config_path.is_file():
                return str(config_path)

        return None

    def _load_json(self, path: Path):
        """加载JSON配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            self._config_data = json.load(f)

    def _load_yaml(self, path: Path):
        """加载YAML配置文件"""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError(
                "需要安装 PyYAML 来读取 YAML 配置文件\n"
                "请运行: uv add pyyaml"
            )

    def _load_toml(self, path: Path):
        """加载TOML配置文件"""
        try:
            import tomli
            with open(path, 'rb') as f:
                self._config_data = tomli.load(f)
        except ImportError:
            raise ImportError(
                "需要安装 tomli 来读取 TOML 配置文件\n"
                "请运行: uv add tomli"
            )

    def load_from_dict(self, config_dict: Dict[str, Any]) -> 'Config':
        """
        从字典加载配置（用于测试或动态配置）

        Args:
            config_dict: 配置字典

        Returns:
            Config实例（支持链式调用）
        """
        self._config_data = config_dict
        self._loaded = True
        # 配置加载成功，静默完成
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套键，懒加载）

        Args:
            key: 配置键，支持嵌套如 "database.host"
            default: 默认值

        Returns:
            配置值

        Examples:
            >>> Config().get("api.zhipuai.key")
            >>> Config().get("database.port", 8000)
        """
        self._ensure_loaded()  # 懒加载

        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_required(self, key: str) -> Any:
        """
        获取必需的配置项（不存在则抛出异常）

        Args:
            key: 配置键

        Returns:
            配置值

        Raises:
            KeyError: 配置项不存在

        Examples:
            >>> Config().get_required("api.zhipuai.key")
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"必需的配置项不存在: {key}")
        return value

    def has(self, key: str) -> bool:
        """
        检查配置项是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """
        return self.get(key) is not None

    def all(self) -> Dict[str, Any]:
        """
        获取所有配置

        Returns:
            配置字典
        """
        self._ensure_loaded()  # 懒加载
        return self._config_data.copy()

    def reload(self) -> 'Config':
        """
        重新加载配置文件

        Returns:
            Config实例
        """
        if not self._config_file:
            raise RuntimeError("没有已加载的配置文件")
        self._loaded = False
        return self._load_from_file(self._config_file)

    @classmethod
    def reset(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None
        cls._config_data = {}
        cls._config_file = None
        cls._loaded = False

    @classmethod
    def is_loaded(cls) -> bool:
        """检查配置是否已加载"""
        return cls._loaded


def load_config(config_file: str = None) -> Config:
    """
    加载配置文件到全局单例（便捷函数）

    如果不指定 config_file，则自动查找配置文件

    Args:
        config_file: 配置文件路径（可选，留空则自动查找）

    Returns:
        Config实例

    Examples:
        >>> # 自动查找并加载（可选，通常不需要手动调用）
        >>> load_config()
        >>>
        >>> # 指定配置文件路径
        >>> load_config("/path/to/config.yaml")
    """
    if config_file is None:
        return Config.load()
    else:
        return Config().load_file(config_file)


def find_config_file() -> Optional[str]:
    """
    从当前工作目录查找配置文件（便捷函数）

    Returns:
        找到的配置文件路径，未找到返回 None

    Examples:
        >>> path = find_config_file()
        >>> if path:
        ...     print(f"找到配置文件: {path}")
    """
    return Config._find_config_file()
