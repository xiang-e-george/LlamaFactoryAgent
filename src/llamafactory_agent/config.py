"""本地控制端配置，与训练镜像中的环境变量分开管理。"""

import json
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """从环境和 .env 读取配置，不在实例化时连接远端或创建目录。

    Args:
        **values: 优先于环境配置的字段值。
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LFA_", extra="ignore", populate_by_name=True
    )

    docker_host: str = Field(
        default="tcp://127.0.0.1:2375",
        validation_alias=AliasChoices("LFA_DOCKER_HOST", "LFA_DOCKER_CONTEXT"),
    )
    docker_image: str = "2f51fed82caa"
    storage_root: str = "/tmp/llamafactory-agent"
    model_scan_dirs: Annotated[list[str], NoDecode] = Field(default_factory=list)
    dataset_source_dirs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["/opt/LlamaFactory/data"]
    )
    runtime_dir: Path = Path(".runtime")
    docker_timeout: int = Field(default=30, ge=1)
    remote_worker_timeout: int = Field(default=120, ge=5)
    poll_interval: float = Field(default=2.0, ge=0.1)
    cleanup_interval: float = Field(default=30.0, ge=0.1)
    failed_container_retention_hours: float = Field(default=0.0, ge=0)
    probe_ttl: float = Field(default=30.0, ge=0)
    max_upload_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8765, ge=1, le=65535)

    @field_validator("model_scan_dirs", "dataset_source_dirs", mode="before")
    @classmethod
    def parse_directories(cls, value: object) -> object:
        """接受 JSON 数组或逗号分隔目录。

        Args:
            value: 环境变量文本或已解析的配置值。
        Returns:
            目录列表；非文本输入保留原值供后续校验。
        """
        if isinstance(value, str):
            return json.loads(value) if value.lstrip().startswith("[") else value.split(",") if value else []
        return value

    @field_validator("storage_root")
    @classmethod
    def validate_storage_root(cls, value: str) -> str:
        """校验专用远端目录，避免将宿主机根目录作为可写挂载。

        Args:
            value: 远端宿主机的绝对目录路径。
        Returns:
            规范化的 Linux 路径。
        """
        path = PurePosixPath(value)
        if not value.startswith("/") or ".." in path.parts or "\\" in value or "\x00" in value:
            raise ValueError("远端存储目录必须是规范的 Linux 绝对路径")
        if str(path) in {"/", "/tmp", "/var", "/home", "/root", "/opt"}:
            raise ValueError("请使用专用子目录作为远端存储目录")
        return str(path)

    @field_validator("model_scan_dirs", "dataset_source_dirs")
    @classmethod
    def validate_directories(cls, values: list[str]) -> list[str]:
        """校验用于只读挂载的目录列表。

        Args:
            values: 远端宿主机目录列表。
        Returns:
            去除首尾空白并规范化后的 Linux 路径列表。
        """
        return [cls.validate_storage_root(value.strip()) for value in values]
