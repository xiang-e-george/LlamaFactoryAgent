"""通过远端 Docker 管理资源和训练的本地 HTTP 接口。"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter, ValidationError
from starlette.types import Receive, Scope, Send

from .schemas import (
    DatasetImportRequest,
    ModelDownloadRequest,
    ResumeRequest,
    TrainingRequest,
    validate_relative_path,
)

IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]
STRING_MAPPING = TypeAdapter(dict[str, str])


def _parse_mapping(value: str, field_name: str) -> dict[str, str]:
    """把上传表单中的字段映射恢复为严格的字符串字典。

    @param value: JSON 表单字段。
    @param field_name: 用于错误提示的字段名。
    @return: 校验通过的字符串映射。
    """
    try:
        return STRING_MAPPING.validate_json(value)
    except ValidationError as error:
        raise HTTPException(422, f"{field_name} 必须是 JSON 字符串字典") from error


def _archive_stream(chunks: Iterator[bytes], first: bytes | None) -> Iterator[bytes]:
    """拼接预读数据和剩余归档，由响应生命周期统一关闭原始流。

    @param chunks: Docker 返回的归档数据流。
    @param first: 响应头发送前已读取的首个数据块。
    @return: 完整归档字节流。
    """
    if first is not None:
        yield first
    for chunk in chunks:
        yield chunk


class ArtifactStreamingResponse(StreamingResponse):
    """在断开连接或发送失败时释放远端归档读取容器。

    Args:
        chunks: 已预读首块的原始归档迭代器。
        first: 预读的数据块，空归档时为 None。
    """

    def __init__(self, chunks: Iterator[bytes], first: bytes | None) -> None:
        super().__init__(
            _archive_stream(chunks, first),
            media_type="application/x-tar",
            headers={"Content-Disposition": 'attachment; filename="artifacts.tar"'},
        )
        self._chunks = chunks

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """确保响应开始前失败或客户端取消时也完成流清理。

        Args:
            scope: ASGI 请求上下文。
            receive: 客户端事件接收函数。
            send: 响应事件发送函数。
        Returns:
            无返回值。
        """
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._chunks, "close", None)
            if close is not None:
                with CancelScope(shield=True):
                    await run_in_threadpool(close)


def create_app(settings: Any = None, service: Any = None) -> FastAPI:
    """创建可注入任务服务的 API，阻塞的 Docker 操作在线程池执行。

    @param settings: 服务配置；省略时读取环境配置。
    @param service: 任务服务；可在测试中注入替代实现。
    @return: 本地 FastAPI 应用。
    """
    from .config import Settings
    from .docker_gateway import GatewayError
    from .service import ServiceError, TrainingService

    settings = settings if settings is not None else Settings()
    service = service if service is not None else TrainingService(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """确保调度器只在应用运行期间工作。

        @param application: 当前 API 应用。
        @return: 应用生命周期上下文。
        """
        await run_in_threadpool(service.start)
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(
        title="LlamaFactory 远端训练 API",
        description=(
            "本地控制端通过 Docker API 调用现有 LlamaFactory 训练。"
            "模型仅从魔搭社区下载；数据集支持上传与登记远端 JSON/JSONL 文件。"
            "耗时操作返回任务 ID，日志使用轮询查询。"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.training_service = service

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Any, error: ServiceError) -> JSONResponse:
        """将业务拒绝转换为稳定的 HTTP 状态。

        @param request: 当前请求。
        @param error: 服务定义的业务错误。
        @return: 带错误原因的 JSON 响应。
        """
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Any, error: GatewayError) -> JSONResponse:
        """向调用方区分远端不可用和参数错误。

        @param request: 当前请求。
        @param error: 远端 Docker 错误。
        @return: 可重试的远端服务错误响应。
        """
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Any, error: ValueError) -> JSONResponse:
        """统一返回参数校验失败。

        @param request: 当前请求。
        @param error: 参数错误。
        @return: 参数错误响应。
        """
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Any, error: KeyError) -> JSONResponse:
        """将缺失资源返回为不存在。

        @param request: 当前请求。
        @param error: 缺失的资源标识。
        @return: 资源不存在响应。
        """
        return JSONResponse(status_code=404, content={"detail": f"资源不存在：{error.args[0]}"})

    @app.get("/health", summary="检查本地 API 状态")
    def health() -> dict[str, str]:
        """本地健康检查不依赖远端 Docker。

        @return: 本地运行状态。
        """
        return {"status": "ok"}

    @app.get("/resources", summary="查询远端 GPU、磁盘和镜像")
    def resources() -> dict:
        """获取远端可用资源。

        @return: GPU、磁盘及镜像信息。
        """
        return service.resources()

    @app.get("/models", summary="查询已下载模型或魔搭下载候选")
    def models(scope: Literal["local", "catalog"] = "local") -> dict:
        """按清单类型查询模型。

        @param scope: 本地模型或镜像提供的下载候选。
        @return: 模型清单。
        """
        return service.models(scope)

    @app.post("/models/refresh", summary="刷新远端模型索引")
    def refresh_models() -> dict:
        """重新探测受管目录中的模型完整性。

        @return: 更新后的模型索引。
        """
        return service.refresh_models()

    @app.post("/models/downloads", status_code=202, summary="从魔搭社区下载模型")
    def download_model(payload: ModelDownloadRequest, idempotency_key: IdempotencyKey = None) -> dict:
        """将魔搭模型下载加入后台队列。

        @param payload: 模型标识与可选版本。
        @param idempotency_key: 防止客户端重试创建重复任务的标识。
        @return: 下载任务。
        """
        return service.download_model(**payload.model_dump(), idempotency_key=idempotency_key)

    @app.get("/datasets", summary="查询已登记的数据集")
    def datasets() -> dict:
        """查询远端可用于训练的数据集。

        @return: 数据集及版本清单。
        """
        return service.datasets()

    @app.get("/datasets/catalog", summary="查询远端现有数据集索引")
    def datasets_catalog() -> dict:
        """读取已有索引，仅本地文件可登记导入。

        @return: 远端数据集索引及来源状态。
        """
        return service.datasets_catalog()

    @app.post("/datasets/imports", status_code=202, summary="登记远端已有文本数据文件")
    def import_dataset(payload: DatasetImportRequest, idempotency_key: IdempotencyKey = None) -> dict:
        """登记远端文件，不接受在线仓库或 URL。

        @param payload: 宿主机绝对路径及数据格式映射。
        @param idempotency_key: 请求幂等标识。
        @return: 数据集导入任务。
        """
        return service.import_dataset(
            **payload.model_dump(), filename=None, local_path=None, idempotency_key=idempotency_key
        )

    @app.post("/datasets/uploads", status_code=202, summary="上传 JSON 或 JSONL 文本数据集")
    def upload_dataset(
        file: Annotated[UploadFile, File()],
        formatting: Annotated[Literal["alpaca", "sharegpt"], Form()] = "alpaca",
        columns: Annotated[str, Form()] = "{}",
        tags: Annotated[str, Form()] = "{}",
        ranking: Annotated[bool, Form()] = False,
        idempotency_key: IdempotencyKey = None,
    ) -> dict:
        """分块暂存上传文件并转交后台导入，失败时清理暂存文件。

        @param file: JSON 或 JSONL 文本文件。
        @param formatting: LlamaFactory 数据格式。
        @param columns: JSON 编码的字段映射。
        @param tags: JSON 编码的对话标签映射。
        @param ranking: 是否为偏好数据。
        @param idempotency_key: 请求幂等标识。
        @return: 数据集导入任务。
        """
        column_mapping = _parse_mapping(columns, "columns")
        tag_mapping = _parse_mapping(tags, "tags")
        filename = PurePosixPath((file.filename or "dataset.json").replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise HTTPException(422, "仅支持 JSON 或 JSONL 文本数据文件")
        upload_dir = Path(settings.runtime_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_path = upload_dir / f"{uuid4().hex}{suffix}"
        try:
            total_bytes = 0
            with local_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_upload_bytes:
                        raise HTTPException(413, "上传文件超过允许的大小")
                    destination.write(chunk)
            if total_bytes == 0:
                raise HTTPException(422, "数据文件不能为空")
            return service.import_dataset(
                source_path=None,
                filename=filename,
                local_path=str(local_path),
                formatting=formatting,
                columns=column_mapping,
                tags=tag_mapping,
                ranking=ranking,
                idempotency_key=idempotency_key,
            )
        except Exception:
            local_path.unlink(missing_ok=True)
            raise
        finally:
            file.file.close()

    @app.get("/datasets/{resource_id}/preview", summary="预览数据集样本")
    def preview_dataset(
        resource_id: str,
        version: str | None = None,
        limit: Annotated[int, Query(ge=1, le=20)] = 3,
    ) -> dict:
        """读取指定版本的有限样本。

        @param resource_id: 数据集标识。
        @param version: 可选数据版本。
        @param limit: 最多返回的样本数。
        @return: 数据集样本。
        """
        return service.preview_dataset(resource_id, version, limit)

    @app.post("/training/preflight", summary="预检模型、数据和训练参数")
    def preflight(payload: TrainingRequest) -> dict:
        """在创建任务前校验训练配置。

        @param payload: 模型、数据集与现有训练参数。
        @return: 配置问题及资源检查结果。
        """
        return service.preflight(**payload.model_dump())

    @app.post("/training/jobs", status_code=202, summary="创建训练任务")
    def create_training(payload: TrainingRequest, idempotency_key: IdempotencyKey = None) -> dict:
        """创建运行现有 LlamaFactory CLI 的任务。

        @param payload: 模型、数据集与现有训练参数。
        @param idempotency_key: 请求幂等标识。
        @return: 训练任务。
        """
        return service.create_training(**payload.model_dump(), idempotency_key=idempotency_key)

    @app.get("/jobs", summary="查询任务列表")
    def jobs() -> dict:
        """统一列出下载、导入与训练任务。

        @return: 任务清单。
        """
        return service.jobs()

    @app.get("/jobs/{job_id}", summary="查询任务状态")
    def job(job_id: str) -> dict:
        """读取任务状态、进度和失败原因。

        @param job_id: 任务标识。
        @return: 任务详情。
        """
        return service.job(job_id)

    @app.get("/jobs/{job_id}/logs", summary="轮询任务日志")
    def logs(job_id: str, tail: Annotated[int, Query(ge=1, le=10000)] = 200) -> dict:
        """返回日志末尾，调用方可以定时轮询。

        @param job_id: 任务标识。
        @param tail: 最多返回的日志行数。
        @return: 任务日志。
        """
        return service.logs(job_id, tail)

    @app.post("/jobs/{job_id}/cancel", summary="取消排队或运行中的任务")
    def cancel(job_id: str) -> dict:
        """停止任务，保留已经生成的检查点与产物。

        @param job_id: 任务标识。
        @return: 更新后的任务状态。
        """
        return service.cancel(job_id)

    @app.get("/training/jobs/{job_id}/artifacts", summary="查询训练产物")
    def artifacts(job_id: str) -> dict:
        """列出任务可下载的产物。

        @param job_id: 训练任务标识。
        @return: 训练产物清单。
        """
        return service.artifacts(job_id)

    @app.get("/training/jobs/{job_id}/artifacts/download", summary="下载训练产物 TAR 归档")
    def download_artifact(job_id: str, path: Annotated[str, Query(min_length=1)]) -> StreamingResponse:
        """在发送响应头前验证产物流，正确报告不存在或远端异常。

        @param job_id: 训练任务标识。
        @param path: 产物目录内的相对路径。
        @return: TAR 格式的产物流。
        """
        relative_path = validate_relative_path(path)
        chunks = iter(service.artifact_archive(job_id, relative_path))
        first = next(chunks, None)
        return ArtifactStreamingResponse(chunks, first)

    @app.post("/training/jobs/{job_id}/resume", status_code=202, summary="从完整检查点创建续训任务")
    def resume(job_id: str, payload: ResumeRequest, idempotency_key: IdempotencyKey = None) -> dict:
        """保留原任务记录并创建新的续训任务。

        @param job_id: 原训练任务标识。
        @param payload: 原任务内的检查点相对路径。
        @param idempotency_key: 请求幂等标识。
        @return: 新的续训任务。
        """
        return service.resume(job_id, payload.checkpoint, idempotency_key=idempotency_key)

    return app
