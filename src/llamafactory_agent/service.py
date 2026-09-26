"""协调资源和任务，远端执行仅通过 DockerGateway。"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator

from filelock import FileLock, Timeout

from .config import Settings
from .docker_gateway import (
    ContainerMissing,
    DockerGateway,
    FileTooLarge,
    GatewayError,
    GatewayUnavailable,
    external_model_path,
)
from .store import TERMINAL_STATES, Store, timestamp

LOGGER = logging.getLogger(__name__)
SERVER_ARGUMENTS = {
    "model_name_or_path",
    "dataset",
    "dataset_dir",
    "output_dir",
    "resume_from_checkpoint",
    "eval_dataset",
    "adapter_name_or_path",
    "cache_dir",
    "tokenized_path",
    "deepspeed",
    "logging_dir",
    "export_dir",
    "hub_token",
    "hf_hub_token",
    "ms_hub_token",
    "push_to_hub",
    "report_to",
    "use_swanlab",
}
DEFAULT_ARGUMENTS = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 0.0001,
    "num_train_epochs": 1,
    "cutoff_len": 1024,
    "logging_steps": 1,
    "save_steps": 100,
    "save_only_model": False,
    "report_to": "none",
    "dataloader_num_workers": 0,
    "preprocessing_num_workers": 1,
}


class ServiceError(ValueError):
    """可直接向调用方返回的业务错误。

    Args:
        message: 面向调用方的错误原因。
        status_code: 对应的 HTTP 状态码，默认 400。
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def request_digest(kind: str, payload: dict) -> str:
    """计算请求摘要，用于检测幂等键被用于不同请求。

    Args:
        kind: 任务类型。
        payload: 参与幂等比较的规范化参数。
    Returns:
        十六进制 SHA-256 摘要。
    """
    content = json.dumps([kind, payload], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def safe_relative_path(value: str) -> str:
    """校验产物路径，拒绝逃逸任务输出目录。

    Args:
        value: 产物目录内的相对路径。
    Returns:
        规范化后的相对路径。
    """
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ServiceError("产物路径必须是输出目录内的相对路径")
    return str(path)


def read_log_tail(path: Path, tail: int) -> str:
    """从文件末尾读取日志，避免每次轮询加载完整训练日志。

    Args:
        path: 已归档的本地日志路径。
        tail: 需要返回的末尾行数，必须大于零。
    Returns:
        按 UTF-8 解码的日志文本。
    """
    if tail < 1:
        raise ValueError("日志行数必须大于零")
    chunks = []
    line_count = 0
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        while position and line_count <= tail:
            size = min(position, 65536)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            line_count += chunk.count(b"\n")
    lines = b"".join(reversed(chunks)).splitlines(keepends=True)
    return b"".join(lines[-tail:]).decode("utf-8", errors="replace")


class TrainingService:
    """单控制端任务服务，容器独立于 API 进程运行。

    Args:
        settings: 本地控制端与远端 Docker 配置。
        gateway: 可注入的 Docker 适配器；省略时根据配置创建。
    """

    def __init__(self, settings: Settings, gateway: DockerGateway | None = None):
        self.settings = settings
        self.gateway = gateway or DockerGateway(settings)
        self.store = Store(settings.runtime_dir / "control.sqlite3")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._guard = threading.RLock()
        self._cache: dict[str, tuple[float, dict]] = {}
        self._storage_ready = False
        self._next_reader_cleanup = 0.0
        self._lock = FileLock(settings.runtime_dir / "controller.lock", thread_local=False)

    def start(self) -> None:
        """启动单调度器；远端不可达不影响本地 API 启动。

        Returns:
            无返回值。
        """
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout as error:
            raise RuntimeError("已有控制端使用该运行目录，请使用单个 API worker") from error
        self.store.initialize()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lfa-scheduler", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """停止本地派发，保留已在远端运行的容器。

        Returns:
            无返回值；调度线程超时未退出时继续持有控制端锁。
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.settings.remote_worker_timeout + self.settings.docker_timeout + 5)
            if self._thread.is_alive():
                LOGGER.warning("调度线程仍在结束远端请求，进程退出前保留控制端锁")
                return
        self.gateway.close()
        self._lock.release()

    def _ensure_storage(self) -> None:
        if not self._storage_ready:
            self.gateway.ensure_storage()
            self._storage_ready = True

    def _cached(self, key: str, loader, ttl: float | None = None) -> dict:
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < (ttl or self.settings.probe_ttl):
            return cached[1]
        result = loader()
        self._cache[key] = (time.monotonic(), result)
        return result

    def resources(self) -> dict:
        """查询远端资源并短时缓存探测结果。

        Returns:
            GPU、磁盘与指定训练镜像的信息。
        """

        def query() -> dict:
            return {**self.gateway.run_worker("resources", {}, gpus=True), "docker": self.gateway.ping()}

        return self._cached("resources", query)

    def models(self, scope: str = "local") -> dict:
        """查询本地资源索引或指定镜像的魔搭模型预设。

        Args:
            scope: local 表示已登记模型，catalog 表示模型预设。
        Returns:
            对应范围的模型清单。
        """
        if scope == "catalog":
            return self._cached("catalog", lambda: self.gateway.run_worker("catalog", {}), ttl=3600)
        if scope != "local":
            raise ServiceError("scope 只支持 local 或 catalog")
        return {"models": self.store.resources("model"), "refresh_endpoint": "/models/refresh"}

    def refresh_models(self) -> dict:
        """扫描受管模型与只读目录，将完整模型登记到本地索引。

        Returns:
            模型扫描结果及成功登记的资源。
        """
        self._ensure_storage()
        roots = ["/lfa/models"] + [external_model_path(path) for path in self.settings.model_scan_dirs]
        mounts = {path: external_model_path(path) for path in self.settings.model_scan_dirs}
        result = self.gateway.run_worker(
            "scan_models", {"roots": roots, "root": "/lfa/models"}, extra_mounts=mounts
        )
        registered = []
        for model in result.get("models", []):
            if model.get("kind") != "model" or not model.get("complete"):
                continue
            resource_id = (
                model.get("resource_id") or "model-" + hashlib.sha256(model["path"].encode()).hexdigest()[:24]
            )
            registered.append(self.store.save_resource("model", resource_id, "1", model))
        return {**result, "registered": registered}

    def download_model(
        self, repo_id: str, revision: str | None = None, idempotency_key: str | None = None
    ) -> dict:
        """提交仅从魔搭下载的模型任务。

        Args:
            repo_id: 魔搭 namespace/model 格式的仓库标识。
            revision: 可选的仓库版本。
            idempotency_key: 防止重复提交的可选请求标识。
        Returns:
            新建或幂等复用的下载任务。
        """
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id) or ".." in repo_id:
            raise ServiceError("请提供魔搭社区 namespace/model 格式的仓库 ID")
        payload = {"repo_id": repo_id, "revision": revision}
        return self._submit("model_download", payload, idempotency_key)

    def datasets(self) -> dict:
        """列出已导入并校验的数据集版本。

        Returns:
            已登记的数据集版本清单。
        """
        return {"datasets": self.store.resources("dataset")}

    def datasets_catalog(self) -> dict:
        """读取允许目录内的数据集索引，跳过所有 Hub 来源。

        Returns:
            本地数据文件清单及被跳过的在线来源数量。
        """
        collected = []
        ignored = 0
        for source in self.settings.dataset_source_dirs:
            result = self.gateway.run_worker(
                "datasets_catalog",
                {"path": "/external/source/dataset_info.json"},
                extra_mounts={source: "/external/source"},
            )
            for entry in result.get("datasets", []):
                relative = PurePosixPath(entry["path"]).relative_to("/external/source")
                collected.append({**entry, "source_path": str(PurePosixPath(source) / relative)})
            ignored += result.get("ignored_online_count", 0)
        return {"datasets": collected, "ignored_online_count": ignored}

    def import_dataset(
        self,
        source_path: str | None = None,
        filename: str | None = None,
        local_path: str | Path | None = None,
        formatting: str = "alpaca",
        columns: dict | None = None,
        tags: dict | None = None,
        ranking: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """提交文件导入任务，生成不可变数据集版本。

        Args:
            source_path: 允许目录内的远端文件，与 local_path 二选一。
            filename: 上传文件的原始文件名。
            local_path: 本地受管上传目录中的暂存文件。
            formatting: alpaca 或 sharegpt 数据格式。
            columns: 数据字段映射。
            tags: 对话角色与内容标签映射。
            ranking: 是否为偏好数据。
            idempotency_key: 防止重复提交的可选请求标识。
        Returns:
            新建或幂等复用的导入任务。
        """
        if bool(source_path) == bool(local_path):
            raise ServiceError("必须且只能选择远端文件或上传文件")
        payload = {"formatting": formatting, "columns": columns or {}, "tags": tags or {}, "ranking": ranking}
        if formatting not in {"alpaca", "sharegpt"}:
            raise ServiceError("数据格式仅支持 alpaca 或 sharegpt")
        if source_path:
            source = PurePosixPath(source_path)
            roots = [PurePosixPath(root) for root in self.settings.dataset_source_dirs]
            roots.append(PurePosixPath(self.settings.storage_root) / "datasets")
            if (
                not source.is_absolute()
                or ".." in source.parts
                or "\\" in source_path
                or not any(source.is_relative_to(root) for root in roots)
            ):
                raise ServiceError("源文件不在配置的远端数据目录内")
            if source.suffix.lower() not in {".json", ".jsonl"}:
                raise ServiceError("首版数据集仅支持 JSON 或 JSONL 文件")
            payload["source_path"] = str(source)
        else:
            upload = Path(local_path).resolve()
            if not upload.is_relative_to((self.settings.runtime_dir / "uploads").resolve()):
                raise ServiceError("上传文件不在本地受管临时目录内")
            if upload.stat().st_size > self.settings.max_upload_bytes:
                raise ServiceError("上传文件超过大小限制", 413)
            if Path(filename or upload.name).suffix.lower() not in {".json", ".jsonl"}:
                raise ServiceError("首版数据集仅支持 JSON 或 JSONL 文件")
            with upload.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            payload.update(local_path=str(upload), filename=filename or upload.name, sha256=checksum)
        job = self._submit("dataset_import", payload, idempotency_key)
        if local_path and job["payload"].get("local_path") != str(Path(local_path).resolve()):
            Path(local_path).unlink(missing_ok=True)
        return job

    def preview_dataset(self, resource_id: str, version: str | None = None, limit: int = 3) -> dict:
        """从远端读取指定数据版本的有限条样本。

        Args:
            resource_id: 数据集资源标识。
            version: 可选版本；省略时读取最新登记版本。
            limit: 样本数量，限制在 1 至 20 条之间。
        Returns:
            样本数组与实际返回的样本数量。
        """
        dataset = self._resource(resource_id, "dataset", version)
        return self.gateway.run_worker(
            "dataset_preview", {"path": dataset["path"], "limit": min(max(limit, 1), 20)}
        )

    def _resource(self, resource_id: str, kind: str, version: str | None = None) -> dict:
        resource = self.store.resource(resource_id, version)
        if resource["kind"] != kind:
            raise ServiceError("资源类型不匹配")
        return resource

    def _training_payload(
        self, model_id: str, dataset_id: str, dataset_version: str | None, training_args: dict
    ) -> dict:
        model = self._resource(model_id, "model")
        dataset = self._resource(dataset_id, "dataset", dataset_version)
        if not model.get("complete"):
            raise ServiceError("模型文件尚未完整下载")
        reserved = SERVER_ARGUMENTS.intersection(training_args)
        if reserved:
            raise ServiceError("以下参数由服务管理：" + ", ".join(sorted(reserved)))
        arguments = {**DEFAULT_ARGUMENTS, **training_args}
        if arguments.get("stage") not in {"sft", "pt", "dpo", "kto", "rm", "ppo"}:
            raise ServiceError("训练阶段不属于现有 LlamaFactory 实现")
        if arguments.get("do_train") is not True:
            raise ServiceError("训练任务需要 do_train=true")
        stage = arguments["stage"]
        if stage in {"rm", "dpo"} and not dataset.get("ranking"):
            raise ServiceError("RM/DPO 训练需要偏好数据集 ranking=true")
        if stage in {"sft", "pt"} and dataset.get("ranking"):
            raise ServiceError("SFT/PT 训练不能使用偏好数据集")
        if not arguments.get("template"):
            raise ServiceError("请指定与模型匹配的 template，可从模型预设查询")
        if any(key.startswith("use_v1") for key in arguments):
            raise ServiceError("不支持新实现相关参数")
        return {"model": model, "dataset": dataset, "training_args": arguments}

    def preflight(
        self,
        model_id: str,
        dataset_id: str,
        dataset_version: str | None = None,
        training_args: dict | None = None,
    ) -> dict:
        """检查资源、训练配置与 GPU；显存信息只作为当前快照。

        Args:
            model_id: 已登记的模型基座标识。
            dataset_id: 已登记的数据集标识。
            dataset_version: 可选数据版本；省略时使用最新版本。
            training_args: 现有 LlamaFactory 训练参数。
        Returns:
            预检是否通过、问题列表、固定的数据版本及资源快照。
        """
        payload = self._training_payload(model_id, dataset_id, dataset_version, training_args or {})
        inspection = self.gateway.run_worker(
            "preflight",
            {
                "model_path": payload["model"]["path"],
                "dataset": payload["dataset"],
                "training_args": payload["training_args"],
            },
            extra_mounts={path: external_model_path(path) for path in self.settings.model_scan_dirs},
        )
        resource_snapshot = self.resources()
        gpus = resource_snapshot.get("gpus", [])
        errors = inspection.get("errors", []) + ([] if gpus else ["远端没有可用的 NVIDIA GPU"])
        return {
            "valid": not errors,
            "errors": errors,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_version": payload["dataset"]["version"],
            "training_args": payload["training_args"],
            "resources": resource_snapshot,
            "notes": ["显存为空闲快照，实际可训练性取决于模型、精度、序列长度及批大小"],
        }

    def create_training(
        self,
        model_id: str,
        dataset_id: str,
        dataset_version: str | None = None,
        training_args: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """固定模型和数据版本后排队，训练启动时重新探测 GPU。

        Args:
            model_id: 已登记的模型基座标识。
            dataset_id: 已登记的数据集标识。
            dataset_version: 可选数据版本；省略时固定当前最新版本。
            training_args: 现有 LlamaFactory 训练参数。
            idempotency_key: 防止重复提交的可选请求标识。
        Returns:
            新建或幂等复用的训练任务。
        """
        payload = self._training_payload(model_id, dataset_id, dataset_version, training_args or {})
        return self._submit("training", payload, idempotency_key)

    def _submit(self, kind: str, payload: dict, key: str | None) -> dict:
        identity = {name: value for name, value in payload.items() if name != "local_path"}
        digest = request_digest(kind, identity)
        with self._guard:
            existing = self.store.by_key(key) if key else None
            if existing:
                if existing["request_hash"] != digest:
                    raise ServiceError("该幂等键已用于不同请求", 409)
                return existing
            job_id = uuid.uuid4().hex
            if kind in {"model_download", "dataset_import"}:
                payload = {**payload, "resource_id": uuid.uuid4().hex, "version": uuid.uuid4().hex}
            return self.store.create_job(job_id, kind, payload, key, digest)

    def jobs(self) -> dict:
        """列出持久化任务记录。

        Returns:
            按提交顺序排列的任务清单。
        """
        return {"jobs": self.store.jobs()}

    def job(self, job_id: str) -> dict:
        """读取任务状态，并为训练任务补充可用的近期指标。

        Args:
            job_id: 任务唯一标识。
        Returns:
            完整任务记录；远端指标不可读时保留本地任务状态。
        """
        job = self.store.job(job_id)
        if job["kind"] == "training" and job["container_id"] and not job["archived_at"]:
            try:
                raw = self.gateway.read_file(
                    job["container_id"], f"/lfa/jobs/{job_id}/output/trainer_log.jsonl"
                )
                complete_lines = raw.splitlines()
                for line in reversed(complete_lines):
                    try:
                        job["metrics"] = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            except FileTooLarge:
                try:
                    result = self._cached(
                        f"metrics:{job_id}:{job['status']}",
                        lambda: self.gateway.run_worker("read_metrics", {"job_id": job_id}),
                        ttl=5,
                    )
                    job["metrics"] = result.get("metrics")
                except GatewayError:
                    job = self.store.job(job_id)
            except GatewayError:
                # 查询期间容器可能已完成归档，返回最新的持久化快照。
                job = self.store.job(job_id)
        return job

    def logs(self, job_id: str, tail: int = 200) -> dict:
        """优先读取本地归档，运行中的任务读取 Docker 日志。

        Args:
            job_id: 任务唯一标识。
            tail: 需要读取的末尾日志行数。
        Returns:
            包含任务标识和日志文本的对象。
        """
        job = self.store.job(job_id)
        path = self._log_path(job_id)
        if path.exists():
            logs = read_log_tail(path, tail)
        elif job["container_id"]:
            try:
                logs = self.gateway.logs(job["container_id"], tail=tail)
            except ContainerMissing:
                # 归档与读取可并发；删容器之前必须已原子发布完整日志。
                if not path.exists():
                    raise
                logs = read_log_tail(path, tail)
        else:
            logs = ""
        return {"job_id": job_id, "logs": logs}

    def cancel(self, job_id: str) -> dict:
        """记录取消意图，由调度器停止运行中的容器。

        Args:
            job_id: 需要取消的任务标识。
        Returns:
            更新后的任务记录；已结束任务保持原状态。
        """
        with self._guard:
            job = self.store.job(job_id)
            if job["status"] in TERMINAL_STATES:
                return job
            status = "cancelled" if job["status"] == "queued" else "cancelling"
            result = self.store.update_job(job_id, cancel_requested=True, status=status)
            if status == "cancelled":
                self._cleanup_upload(result)
            return result

    def artifacts(self, job_id: str) -> dict:
        """读取本任务输出目录内的文件元信息。

        Args:
            job_id: 训练任务唯一标识。
        Returns:
            可供下载的产物清单。
        """
        job = self.store.job(job_id)
        if job["kind"] != "training":
            raise ServiceError("只有训练任务具有模型产物")
        return self.gateway.run_worker("list_artifacts", {"job_id": job_id})

    def artifact_archive(self, job_id: str, relative_path: str) -> Iterator[bytes]:
        """以 tar 流下载明确列出的产物，拒绝目录逃逸。

        Args:
            job_id: 训练任务唯一标识。
            relative_path: 产物清单中的相对路径。
        Returns:
            包含指定文件或目录的 tar 字节迭代器。
        """
        path = safe_relative_path(relative_path)
        listing = self.artifacts(job_id).get("artifacts", [])
        if not any(item["path"] == path for item in listing):
            raise ServiceError("产物不存在", 404)
        return self.gateway.artifact_archive(job_id, path)

    def resume(self, job_id: str, checkpoint: str, idempotency_key: str | None = None) -> dict:
        """从已结束任务的完整检查点创建独立续训任务。

        Args:
            job_id: 原训练任务标识。
            checkpoint: checkpoint-步数 格式的检查点名称。
            idempotency_key: 防止重复提交的可选请求标识。
        Returns:
            新建或幂等复用的续训任务。
        """
        original = self.store.job(job_id)
        if original["kind"] != "training" or original["status"] not in TERMINAL_STATES:
            raise ServiceError("只能从已结束的训练任务续训", 409)
        checkpoint_name = safe_relative_path(checkpoint)
        if not re.fullmatch(r"checkpoint-\d+", checkpoint_name):
            raise ServiceError("请提供 checkpoint-步数 格式的检查点名称")
        path = f"/lfa/jobs/{job_id}/output/{checkpoint_name}"
        inspected = self.gateway.run_worker("check_checkpoint", {"path": path})
        if not inspected.get("valid"):
            raise ServiceError("检查点不完整：" + ", ".join(inspected.get("missing", [])))
        payload = {**original["payload"], "resume_from_checkpoint": path, "parent_job_id": job_id}
        return self._submit("training", payload, idempotency_key)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                LOGGER.exception("任务调度发生错误，下一轮继续对账")
            self._stop.wait(self.settings.poll_interval)

    def tick(self) -> None:
        """对账远端容器，分别从训练与导入队列派发任务。

        Returns:
            无返回值；状态变化写入任务记录。
        """
        for job in self.store.jobs(active_only=True):
            if job["status"] != "queued":
                try:
                    self._reconcile(job)
                except GatewayUnavailable as error:
                    self.store.update_job(job["id"], error=f"远端暂不可达：{error}")
                except ContainerMissing:
                    self.store.update_job(job["id"], status="failed", error="远端任务容器已被移除")
                    self._cleanup_upload(job)
                except (GatewayError, ServiceError, OSError) as error:
                    self.store.update_job(job["id"], status="failed", error=str(error))
                    self._cleanup_upload(job)
        active = self.store.jobs(active_only=True)
        for lane in ("training", "transfer"):
            lane_jobs = [job for job in active if (job["kind"] == "training") == (lane == "training")]
            if any(job["status"] != "queued" for job in lane_jobs):
                continue
            queued = next(iter(lane_jobs), None)
            if queued and not self._stop.is_set():
                try:
                    self._launch(queued)
                except GatewayUnavailable as error:
                    self.store.update_job(queued["id"], error=f"远端暂不可达：{error}")
                except (GatewayError, ServiceError, OSError) as error:
                    self.store.update_job(queued["id"], status="failed", error=str(error))
                    self._cleanup_upload(queued)
        self.cleanup_finished()

    def cleanup_finished(self) -> None:
        """归档终态任务并回收容器，失败时持久化重试时间。

        Returns:
            无返回值；归档和回收错误独立记录，不改变训练结果。
        """
        for job in self.store.pending_cleanup():
            try:
                self._cleanup_finished_job(job)
            except (GatewayError, OSError) as error:
                retry_at = datetime.now(timezone.utc) + timedelta(seconds=self.settings.cleanup_interval)
                self.store.update_job(
                    job["id"], cleanup_error=str(error), cleanup_next_attempt_at=retry_at.isoformat()
                )
                LOGGER.warning("任务 %s 回收未完成，稍后重试：%s", job["id"], error)
        if time.monotonic() >= self._next_reader_cleanup:
            self._next_reader_cleanup = time.monotonic() + self.settings.cleanup_interval
            try:
                self.gateway.cleanup_artifact_readers()
            except GatewayError as error:
                LOGGER.warning("临时产物读取容器回收未完成，稍后重试：%s", error)

    def _log_path(self, job_id: str) -> Path:
        return self.settings.runtime_dir / "jobs" / job_id / "console.log"

    def _archive_finished_job(self, job: dict) -> dict:
        state = self.gateway.inspect_job(job["container_id"], job["id"])
        if state["running"] or state["status"] not in {"created", "exited", "dead"}:
            raise GatewayError("任务容器尚未停止，暂不归档或回收")
        path = self._log_path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".log.partial")
        chunks = self.gateway.stream_logs(job["container_id"])
        try:
            with partial.open("wb") as handle:
                for chunk in chunks:
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            partial.replace(path)
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
            partial.unlink(missing_ok=True)
        metrics = None
        if job["kind"] == "training":
            metrics = self.gateway.run_worker("read_metrics", {"job_id": job["id"]}).get("metrics")
        return self.store.update_job(
            job["id"],
            archived_at=timestamp(),
            metrics=metrics,
            exit_code=state["exit_code"] if state["status"] != "created" else None,
            cleanup_error=None,
        )

    def _cleanup_finished_job(self, job: dict) -> None:
        if not job["archived_at"] or not self._log_path(job["id"]).is_file():
            try:
                job = self._archive_finished_job(job)
            except ContainerMissing:
                self.store.update_job(
                    job["id"],
                    container_removed_at=timestamp(),
                    archived_at=None,
                    cleanup_error="容器在归档完成前已被移除，无法补齐归档；已保存的记录和日志保留",
                    cleanup_next_attempt_at=None,
                )
                return
        if job["status"] == "failed" and self.settings.failed_container_retention_hours:
            deadline = datetime.fromisoformat(job["finished_at"]) + timedelta(
                hours=self.settings.failed_container_retention_hours
            )
            if datetime.now(timezone.utc) < deadline:
                self.store.update_job(job["id"], cleanup_next_attempt_at=deadline.isoformat())
                return
        try:
            self.gateway.remove_job(job["container_id"], job["id"])
        except ContainerMissing:
            # 删除响应丢失后再次对账，已归档且不存在等价于回收完成。
            pass
        self.store.update_job(
            job["id"], container_removed_at=timestamp(), cleanup_error=None, cleanup_next_attempt_at=None
        )

    def _launch(self, job: dict) -> None:
        with self._guard:
            if self.store.job(job["id"])["status"] != "queued":
                return
            self.store.update_job(job["id"], status="starting", error=None)
        self._ensure_storage()
        self._create_and_start(self.store.job(job["id"]))

    def _create_and_start(self, job: dict) -> None:
        container_id = job["container_id"] or self.gateway.find_job(job["id"])
        payload = job["payload"]
        if not container_id:
            if job["kind"] == "training":
                hardware = self.gateway.run_worker("resources", {}, gpus=True)
                if not hardware.get("gpus"):
                    raise ServiceError("训练启动失败：远端未发现 GPU")
                self.gateway.run_worker(
                    "prepare_train",
                    {
                        "job_id": job["id"],
                        "model_path": payload["model"]["path"],
                        "dataset": payload["dataset"],
                        "training_args": payload["training_args"],
                        "resume_from_checkpoint": payload.get("resume_from_checkpoint"),
                    },
                    extra_mounts={path: external_model_path(path) for path in self.settings.model_scan_dirs},
                )
                container_id = self.gateway.create_training(job["id"], hardware["gpus"][0]["uuid"])
            else:
                worker_payload = dict(payload)
                mounts = {}
                if job["kind"] == "dataset_import":
                    if payload.get("source_path"):
                        source = PurePosixPath(payload["source_path"])
                        mounts[str(source.parent)] = "/external/source"
                        worker_payload["source_path"] = f"/external/source/{source.name}"
                    else:
                        suffix = Path(payload["filename"]).suffix.lower()
                        worker_payload["source_path"] = f"/lfa/incoming/{job['id']}/upload{suffix}"
                container_id = self.gateway.create_worker(
                    job["id"], job["kind"], worker_payload, extra_mounts=mounts
                )
            self.store.update_job(job["id"], container_id=container_id)
        state = self.gateway.inspect(container_id)
        if state["status"] == "created":
            if job["kind"] == "dataset_import" and payload.get("local_path"):
                suffix = Path(payload["filename"]).suffix.lower()
                self.gateway.upload_file(
                    container_id, Path(payload["local_path"]), f"incoming/{job['id']}/upload{suffix}"
                )
            if self.store.job(job["id"])["cancel_requested"]:
                self.store.update_job(job["id"], status="cancelled")
                self._cleanup_upload(job)
                return
            self.gateway.start(container_id)
        self.store.update_job(job["id"], status="running", error=None, container_id=container_id)

    def _reconcile(self, job: dict) -> None:
        container_id = job["container_id"] or self.gateway.find_job(job["id"])
        if not container_id:
            if job["cancel_requested"]:
                self.store.update_job(job["id"], status="cancelled")
                self._cleanup_upload(job)
                return
            self._ensure_storage()
            self._create_and_start(job)
            return
        if not job["container_id"]:
            self.store.update_job(job["id"], container_id=container_id)
            job["container_id"] = container_id
        state = self.gateway.inspect(container_id)
        if job["cancel_requested"]:
            if state["running"]:
                self.gateway.stop(container_id)
            self.store.update_job(job["id"], status="cancelled", error=None)
            self._cleanup_upload(job)
            return
        if state["status"] == "created":
            self._create_and_start(job)
        elif state["running"]:
            self.store.update_job(job["id"], status="running", error=None)
        elif state["status"] in {"exited", "dead"}:
            self._finish(job, state)

    def _finish(self, job: dict, state: dict) -> None:
        if state["exit_code"] != 0 or state.get("oom_killed"):
            detail = "容器内存不足" if state.get("oom_killed") else f"容器退出码 {state['exit_code']}"
            tail = self.gateway.logs(job["container_id"], tail=40)
            self.store.update_job(job["id"], status="failed", error=f"{detail}\n{tail}"[-12000:])
        elif job["kind"] == "training":
            result = self.artifacts(job["id"])
            files = {entry["path"] for entry in result.get("artifacts", [])}
            saved = any(
                path in files
                for path in (
                    "adapter_model.safetensors",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                )
            )
            if not saved:
                self.store.update_job(
                    job["id"], status="failed", result=result, error="容器正常退出，但未发现最终模型权重"
                )
            else:
                self.store.update_job(job["id"], status="succeeded", result=result, error=None)
        else:
            result = self.gateway.worker_result(job["container_id"])
            if job["kind"] == "model_download":
                if not result.get("complete"):
                    raise GatewayError("模型下载结束，但文件完整性校验未通过")
                resource = self.store.save_resource("model", job["payload"]["resource_id"], "1", result)
            else:
                resource = self.store.save_resource(
                    "dataset", job["payload"]["resource_id"], job["payload"]["version"], result
                )
            self.store.update_job(job["id"], status="succeeded", result=resource, error=None)
        self._cleanup_upload(job)

    @staticmethod
    def _cleanup_upload(job: dict) -> None:
        upload = job["payload"].get("local_path")
        if upload:
            Path(upload).unlink(missing_ok=True)
