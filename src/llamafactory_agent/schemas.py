"""本地训练 API 的请求约束。"""

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestModel(BaseModel):
    """拒绝未约定字段，避免调用方误以为参数已生效。"""

    model_config = ConfigDict(extra="forbid")


class ModelDownloadRequest(RequestModel):
    """魔搭社区模型下载请求。"""

    repo_id: str = Field(
        min_length=3,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="魔搭社区模型标识，例如 Qwen/Qwen2.5-0.5B-Instruct",
    )
    revision: str | None = Field(default=None, min_length=1, max_length=256)


class DatasetImportRequest(RequestModel):
    """登记远端宿主机已有的文本数据文件。"""

    source_path: str = Field(description="远端宿主机 JSON 或 JSONL 文件的绝对路径")
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    columns: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    ranking: bool = False

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """限定远端文件路径，避免将 URL 或模糊路径作为数据源。

        @param value: 远端宿主机文件路径。
        @return: 校验通过的原始路径。
        """
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
            raise ValueError("source_path 必须是不含 .. 或反斜杠的 Linux 绝对路径")
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("仅支持 JSON 或 JSONL 文本数据文件")
        return value


class TrainingRequest(RequestModel):
    """固定模型与数据集版本后提交现有 LlamaFactory 训练参数。"""

    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = Field(default=None, min_length=1)
    training_args: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(RequestModel):
    """从任务的完整检查点创建续训任务。"""

    checkpoint: str = Field(min_length=1, description="任务产物内的检查点相对路径")

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(cls, value: str) -> str:
        """保证检查点引用不越过原任务目录。

        @param value: 检查点的相对路径。
        @return: 校验通过的相对路径。
        """
        return validate_relative_path(value)


def validate_relative_path(value: str) -> str:
    """限制产物引用在任务目录内，跨平台路径均不能逃逸。

    @param value: 任务目录内的非空相对路径。
    @return: 校验通过的原始路径。
    """
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError("路径必须是任务目录内不含 .. 的相对路径")
    return value
