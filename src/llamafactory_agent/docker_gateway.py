"""通过 Docker API 操作远端任务，避免依赖 SSH 或额外常驻服务。"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

import docker
from docker.errors import APIError, DockerException, NotFound
from docker.types import DeviceRequest, Mount
from requests.exceptions import RequestException

if TYPE_CHECKING:
    from .config import Settings


LOGGER = logging.getLogger(__name__)
RESULT_PREFIX = "LFA_RESULT="
READ_ONLY_ACTIONS = {
    "resources",
    "catalog",
    "scan_models",
    "datasets_catalog",
    "dataset_preview",
    "list_artifacts",
    "check_checkpoint",
    "preflight",
    "read_metrics",
}
Result = TypeVar("Result")


def external_model_path(host_path: str) -> str:
    """将宿主机模型目录转换为与配置顺序无关的稳定挂载路径。

    Args:
        host_path: 已验证的远端宿主机模型目录。
    Returns:
        位于 /external/models 下的容器挂载路径。
    """
    normalized = str(PurePosixPath(host_path))
    directory_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"/external/models/{directory_id}"


class GatewayError(RuntimeError):
    """远端操作失败，但不代表 Docker 连接不可用。"""


class GatewayUnavailable(GatewayError):
    """远端暂时不可观测，调用方应保留任务原有状态。"""


class ContainerMissing(GatewayError):
    """指定容器已不存在，需要调用方重新对账。"""


class FileTooLarge(GatewayError):
    """文件超出直接读取上限，调用方可改用远端有限尾读。"""


def _invoke(operation: Callable[[], Result], *, container_lookup: bool = False) -> Result:
    """统一远端错误语义，防止网络中断被误判为训练失败。"""
    try:
        return operation()
    except NotFound as error:
        exception_type = ContainerMissing if container_lookup else GatewayError
        raise exception_type(str(error)) from error
    except APIError as error:
        exception_type = GatewayUnavailable if error.status_code in {502, 503, 504} else GatewayError
        raise exception_type(str(error)) from error
    except (RequestException, DockerException, OSError) as error:
        raise GatewayUnavailable(f"Docker 连接暂时不可用：{error}") from error


class _ChunkReader(io.RawIOBase):
    """将 Docker 分块响应适配为 tarfile 可逐段消费的字节流。"""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._pending = bytearray()

    def readable(self) -> bool:
        return True

    def readinto(self, target: bytearray) -> int:
        while not self._pending:
            chunk = next(self._chunks, None)
            if chunk is None:
                return 0
            self._pending.extend(chunk)
        count = min(len(target), len(self._pending))
        target[:count] = self._pending[:count]
        del self._pending[:count]
        return count

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close:
            close()
        super().close()


class DockerGateway:
    """封装固定镜像、受管挂载和任务容器的完整生命周期。

    Args:
        settings: Docker 连接、镜像和远端存储配置。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: docker.DockerClient | None = None
        self._connection_lock = threading.RLock()
        self._image_id: str | None = None
        self._reader_owner = uuid.uuid4().hex
        scope = [
            str(Path(settings.runtime_dir).resolve()),
            settings.docker_host,
            str(PurePosixPath(settings.storage_root)),
        ]
        self._reader_scope = hashlib.sha256(json.dumps(scope).encode("utf-8")).hexdigest()
        self._reader_lock = threading.Lock()
        self._active_readers: set[str] = set()

    @property
    def client(self) -> docker.DockerClient:
        """按需建立 Docker 连接，允许本地 API 在远端离线时启动。"""
        with self._connection_lock:
            if self._client is None:
                self._client = _invoke(
                    lambda: docker.DockerClient(
                        base_url=self.settings.docker_host,
                        version="auto",
                        timeout=self.settings.docker_timeout,
                    )
                )
            return self._client

    def close(self) -> None:
        """关闭本地连接池，不影响远端已启动的容器。"""
        if self._client is not None:
            self._client.close()

    def _image(self) -> str:
        with self._connection_lock:
            if self._image_id is None:
                image = _invoke(lambda: self.client.images.get(self.settings.docker_image))
                self._image_id = image.id
            return self._image_id

    def ping(self) -> dict[str, Any]:
        """查询 Docker 和固定镜像信息。

        Returns:
            Docker 版本及完整镜像 ID、标签。
        """
        _invoke(self.client.ping)
        version = _invoke(self.client.version)
        image = _invoke(lambda: self.client.images.get(self._image()))
        return {
            "docker_version": version.get("Version"),
            "api_version": version.get("ApiVersion"),
            "image_id": image.id,
            "image_tags": image.tags,
        }

    def _mounts(self, *, readonly: bool, extra_mounts: dict[str, str] | None = None) -> list[Mount]:
        mounts = [Mount("/lfa", self.settings.storage_root, type="bind", read_only=readonly)]
        for host_path, container_path in (extra_mounts or {}).items():
            mounts.append(Mount(container_path, host_path, type="bind", read_only=True))
        return mounts

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "USE_V1": "0",
            "USE_MODELSCOPE_HUB": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "MODELSCOPE_CACHE": "/lfa/cache/modelscope",
        }

    @staticmethod
    def _put_bytes(container: Any, directory: str, filename: str, content: bytes) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            info = tarfile.TarInfo(filename)
            info.size = len(content)
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(content))
        archive.seek(0)
        accepted = _invoke(lambda: container.put_archive(directory, archive))
        if not accepted:
            raise GatewayError(f"Docker 未接受上传文件：{filename}")

    def _install_worker_source(self, container: Any) -> None:
        worker_source = Path(__file__).with_name("remote_worker.py").read_bytes()
        self._put_bytes(container, "/tmp", "lfa-worker.py", worker_source)

    def _create_worker(
        self,
        name: str,
        action: str,
        payload: dict[str, Any],
        *,
        labels: dict[str, str],
        gpus: bool = False,
        extra_mounts: dict[str, str] | None = None,
    ) -> Any:
        options: dict[str, Any] = {
            "image": self._image(),
            "name": name,
            "command": [
                "python",
                "-B",
                "/tmp/lfa-worker.py",
                action,
                json.dumps(payload, ensure_ascii=False),
            ],
            "entrypoint": [],
            "environment": self._environment(),
            "labels": {**labels, "lfa.worker": "true"},
            "init": True,
            "restart_policy": {"Name": "no"},
        }
        if action == "init":
            # 仅初始化允许 Docker 创建宿主机根目录，查询不能隐式改变目录。
            options["volumes"] = {self.settings.storage_root: {"bind": "/lfa", "mode": "rw"}}
        elif action == "resources":
            storage_parent = str(PurePosixPath(self.settings.storage_root).parent)
            options["mounts"] = [Mount("/lfa", storage_parent, type="bind", read_only=True)]
        elif action == "datasets_catalog":
            options["mounts"] = [
                Mount(container_path, host_path, type="bind", read_only=True)
                for host_path, container_path in (extra_mounts or {}).items()
            ]
        elif action != "catalog":
            options["mounts"] = self._mounts(readonly=action in READ_ONLY_ACTIONS, extra_mounts=extra_mounts)
        if gpus:
            options["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        container = _invoke(lambda: self.client.containers.create(**options))
        try:
            self._install_worker_source(container)
        except Exception:
            self._cleanup(container)
            raise
        return container

    def _cleanup(self, container: Any) -> None:
        try:
            _invoke(lambda: container.remove(force=True))
        except GatewayError as error:
            LOGGER.warning("临时容器 %s 暂未清理：%s", container.id, error)

    def run_worker(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        gpus: bool = False,
        extra_mounts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """执行并清理一次性工具容器。

        Args:
            action: 内置远端操作名称。
            payload: 传递给操作的 JSON 参数。
            gpus: 是否向探测容器开放全部 GPU。
            extra_mounts: 宿主机目录到容器目录的只读映射。
        Returns:
            工具输出的结构化结果。
        """
        container = self._create_worker(
            f"lfa-tool-{uuid.uuid4().hex}",
            action,
            payload,
            labels={"lfa.managed": "true", "lfa.tool": "true"},
            gpus=gpus,
            extra_mounts=extra_mounts,
        )
        try:
            _invoke(container.start)
            result = _invoke(lambda: container.wait(timeout=self.settings.remote_worker_timeout))
            output = _invoke(lambda: container.logs(stdout=True, stderr=True)).decode(
                "utf-8", errors="replace"
            )
            if result["StatusCode"] != 0:
                raise GatewayError(f"远端操作 {action} 失败：\n{output[-8000:]}")
            return self._parse_result(output)
        finally:
            self._cleanup(container)

    def create_worker(
        self,
        job_id: str,
        action: str,
        payload: dict[str, Any],
        extra_mounts: dict[str, str] | None = None,
    ) -> str:
        """创建后台任务容器并传入程序，不启动任务。

        Args:
            job_id: 任务唯一标识。
            action: 内置后台操作名称。
            payload: 传递给操作的 JSON 参数。
            extra_mounts: 宿主机目录到容器目录的只读映射。
        Returns:
            创建的容器 ID。
        """
        container = self._create_worker(
            f"lfa-{job_id}",
            action,
            payload,
            labels={"lfa.managed": "true", "lfa.job_id": job_id},
            extra_mounts=extra_mounts,
        )
        return container.id

    def create_training(self, job_id: str, gpu_uuid: str) -> str:
        """创建仅绑定指定 GPU 的旧版训练容器。

        Args:
            job_id: 已生成配置的训练任务标识。
            gpu_uuid: 预检选定的 GPU UUID。
        Returns:
            尚未启动的容器 ID。
        """
        extra_mounts = {
            directory: external_model_path(directory) for directory in self.settings.model_scan_dirs
        }
        mounts = self._mounts(readonly=True, extra_mounts=extra_mounts)
        mounts.append(
            Mount(
                f"/lfa/jobs/{job_id}",
                str(PurePosixPath(self.settings.storage_root) / "jobs" / job_id),
                type="bind",
                read_only=False,
            )
        )
        cache_root = f"/lfa/jobs/{job_id}/cache"
        environment = {
            **self._environment(),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": f"{cache_root}/huggingface",
            "HF_DATASETS_CACHE": f"{cache_root}/datasets",
            "MODELSCOPE_CACHE": f"{cache_root}/modelscope",
        }
        container = _invoke(
            lambda: self.client.containers.create(
                image=self._image(),
                name=f"lfa-{job_id}",
                command=["llamafactory-cli", "train", f"/lfa/jobs/{job_id}/train.yaml"],
                environment=environment,
                labels={"lfa.managed": "true", "lfa.job_id": job_id},
                mounts=mounts,
                device_requests=[DeviceRequest(device_ids=[gpu_uuid], capabilities=[["gpu"]])],
                init=True,
                shm_size="2g",
                restart_policy={"Name": "no"},
            )
        )
        return container.id

    def _container(self, container_id: str) -> Any:
        return _invoke(lambda: self.client.containers.get(container_id), container_lookup=True)

    def start(self, container_id: str) -> None:
        """启动已创建的容器，参数为容器 ID。"""
        container = self._container(container_id)
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        state = container.attrs.get("State", {}).get("Status")
        if labels.get("lfa.worker") == "true" and state == "created":
            # 创建响应或上传响应丢失后，恢复不能假定容器中已存在完整程序。
            self._install_worker_source(container)
        _invoke(container.start)

    def inspect(self, container_id: str) -> dict[str, Any]:
        """读取容器实际状态，返回归一化状态、退出码和失败原因。"""
        container = self._container(container_id)
        return self._state(container)

    @staticmethod
    def _state(container: Any) -> dict[str, Any]:
        state = container.attrs["State"]
        return {
            "status": state["Status"],
            "running": state["Running"],
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled", False),
            "error": state.get("Error", ""),
        }

    def inspect_job(self, container_id: str, job_id: str) -> dict[str, Any]:
        """核验任务归属与存储挂载后读取容器状态。

        Args:
            container_id: 本地任务记录保存的容器 ID。
            job_id: 同一条任务记录的唯一标识。
        Returns:
            与 inspect 相同的归一化容器状态。
        """
        container = self._container(container_id)
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        if labels.get("lfa.managed") != "true" or labels.get("lfa.job_id") != job_id:
            raise GatewayError("容器标签与任务记录不匹配，拒绝访问或删除")
        storage_root = str(PurePosixPath(self.settings.storage_root))
        matching_mount = any(
            mount.get("Type") == "bind"
            and mount.get("Destination") == "/lfa"
            and str(PurePosixPath(mount.get("Source", ""))) == storage_root
            for mount in container.attrs.get("Mounts", [])
        )
        if not matching_mount:
            raise GatewayError("容器的 /lfa 挂载不属于当前存储目录，拒绝访问或删除")
        return self._state(container)

    def find_job(self, job_id: str) -> str | None:
        """通过任务标签查找容器，返回容器 ID 或空值。"""
        containers = _invoke(
            lambda: self.client.containers.list(
                all=True,
                filters={"label": ["lfa.managed=true", f"lfa.job_id={job_id}"]},
            )
        )
        return containers[0].id if containers else None

    def logs(self, container_id: str, tail: int = 200) -> str:
        """读取容器末尾日志，参数为容器 ID 和日志行数。"""
        container = self._container(container_id)
        content = _invoke(lambda: container.logs(stdout=True, stderr=True, tail=tail), container_lookup=True)
        return content.decode("utf-8", errors="replace")

    def stream_logs(self, container_id: str) -> Iterator[bytes]:
        """有限流式读取现有完整日志，不等待后续日志产生。

        Args:
            container_id: 需要归档日志的容器 ID。
        Yields:
            包含标准输出与错误输出的日志字节块。
        """
        container = self._container(container_id)
        if container.attrs.get("State", {}).get("Status") == "created":
            return
        chunks = _invoke(lambda: container.logs(stdout=True, stderr=True, stream=True, follow=False))
        yield from self._consume_stream(chunks)

    def stop(self, container_id: str, timeout: int = 20) -> None:
        """停止指定容器，等待秒数由 timeout 指定。"""
        container = self._container(container_id)
        _invoke(lambda: container.stop(timeout=timeout))

    def remove(self, container_id: str) -> None:
        """删除已停止的容器，保留宿主机挂载目录中的产物。"""
        _invoke(self._container(container_id).remove)

    def remove_job(self, container_id: str, job_id: str) -> None:
        """删除归属确认且已停止的任务容器，保留所有挂载数据。

        Args:
            container_id: 本地任务记录保存的容器 ID。
            job_id: 同一条任务记录的唯一标识。
        """
        state = self.inspect_job(container_id, job_id)
        if state["running"] or state["status"] not in {"created", "exited", "dead"}:
            raise GatewayError("任务容器尚未停止，拒绝清理")
        # 不使用 force；若状态在核验后发生变化，由 Docker 拒绝删除运行容器。
        _invoke(self._container(container_id).remove, container_lookup=True)

    def archive(self, container_id: str, path: str) -> Iterator[bytes]:
        """流式读取容器路径的原始 tar 包，返回字节块迭代器。"""
        container = self._container(container_id)
        yield from self._archive_from_container(container, path)

    def _archive_from_container(self, container: Any, path: str) -> Iterator[bytes]:
        chunks, _ = _invoke(lambda: container.get_archive(path, chunk_size=65536))
        yield from self._consume_stream(chunks)

    @staticmethod
    def _consume_stream(chunks: Iterator[bytes]) -> Iterator[bytes]:
        try:
            while True:
                chunk = _invoke(lambda: next(chunks, None))
                if chunk is None:
                    return
                yield chunk
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()

    def artifact_archive(self, job_id: str, relative_path: str) -> Iterator[bytes]:
        """通过临时只读容器下载产物，不依赖原训练容器。

        Args:
            job_id: 产物所属的任务标识。
            relative_path: 已由服务层确认存在的产物相对路径。
        Yields:
            Docker 返回的原始 tar 字节块；结束或关闭时清理临时容器。
        """
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id):
            raise ValueError("无效的产物任务标识")
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or path.is_absolute()
            or ".." in path.parts
            or any(character in relative_path for character in ("\\", "\x00", ":"))
        ):
            raise ValueError("产物路径必须位于任务输出目录内")
        output = str(PurePosixPath(self.settings.storage_root) / "jobs" / job_id / "output")
        options = {
            "image": self._image(),
            "name": f"lfa-artifact-{uuid.uuid4().hex}",
            "command": ["true"],
            "entrypoint": [],
            "labels": {
                "lfa.managed": "true",
                "lfa.artifact_reader": "true",
                "lfa.artifact_job_id": job_id,
                "lfa.storage_root": str(PurePosixPath(self.settings.storage_root)),
                "lfa.controller_scope": self._reader_scope,
                "lfa.owner": self._reader_owner,
            },
            "mounts": [Mount("/output", output, type="bind", read_only=True)],
            "restart_policy": {"Name": "no"},
        }
        with self._reader_lock:
            # 创建和登记必须连续，回收器才能区分正在建立的下载与应答丢失后的孤儿。
            container = _invoke(lambda: self.client.containers.create(**options))
            self._active_readers.add(container.id)
        try:
            yield from self._archive_from_container(container, f"/output/{path}")
        finally:
            try:
                _invoke(container.remove, container_lookup=True)
            except ContainerMissing:
                pass
            except GatewayError as error:
                LOGGER.warning("产物读取容器 %s 暂未清理：%s", container.id, error)
            finally:
                with self._reader_lock:
                    self._active_readers.discard(container.id)

    def cleanup_artifact_readers(self) -> int:
        """回收同一控制端遗留的读取容器，跳过当前仍在传输的容器。

        Returns:
            本轮实际删除的容器数量；连接异常由调用方下轮重试。
        """
        containers = _invoke(
            lambda: self.client.containers.list(
                all=True,
                filters={
                    "label": [
                        "lfa.managed=true",
                        "lfa.artifact_reader=true",
                        f"lfa.controller_scope={self._reader_scope}",
                    ]
                },
            )
        )
        removed = 0
        for candidate in containers:
            try:
                container = self._container(candidate.id)
                with self._reader_lock:
                    if container.id in self._active_readers:
                        continue
                if not self._owned_artifact_reader(container):
                    continue
                state = self._state(container)
                if state["running"] or state["status"] not in {"created", "exited", "dead"}:
                    continue
                _invoke(container.remove, container_lookup=True)
                removed += 1
            except ContainerMissing:
                pass
        return removed

    def _owned_artifact_reader(self, container: Any) -> bool:
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        job_id = labels.get("lfa.artifact_job_id", "")
        storage_root = str(PurePosixPath(self.settings.storage_root))
        if (
            labels.get("lfa.managed") != "true"
            or labels.get("lfa.artifact_reader") != "true"
            or labels.get("lfa.controller_scope") != self._reader_scope
            or labels.get("lfa.storage_root") != storage_root
            or not labels.get("lfa.owner")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id)
        ):
            return False
        output = str(PurePosixPath(storage_root) / "jobs" / job_id / "output")
        return any(
            mount.get("Type") == "bind"
            and mount.get("Destination") == "/output"
            and mount.get("Source") == output
            and mount.get("RW") is False
            for mount in container.attrs.get("Mounts", [])
        )

    def read_file(self, container_id: str, path: str, max_bytes: int = 2 * 1024 * 1024) -> bytes:
        """读取单个普通文件，并限制内存占用。

        Args:
            container_id: 容器 ID，允许容器处于停止状态。
            path: 调用方已验证的容器内文件路径。
            max_bytes: 允许的最大文件大小。
        Returns:
            文件原始内容。
        """
        with io.BufferedReader(_ChunkReader(self.archive(container_id, path))) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as bundle:
                member = bundle.next()
                if member is None or not member.isfile():
                    raise GatewayError("请求路径不是普通文件")
                if member.size > max_bytes:
                    raise FileTooLarge(f"文件超过读取上限 {max_bytes} 字节")
                file = bundle.extractfile(member)
                if file is None:
                    raise GatewayError("无法读取归档中的文件")
                return file.read()

    def upload_file(self, container_id: str, local_path: str | Path, remote_relative_path: str) -> None:
        """将本地文件上传到受管根目录，允许目标容器尚未启动。

        Args:
            container_id: 具有可写 /lfa 挂载的容器 ID。
            local_path: 本地源文件。
            remote_relative_path: /lfa 下无跳转段的相对路径。
        """
        parts = remote_relative_path.split("/")
        if "\\" in remote_relative_path or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("上传目标必须是 /lfa 下的规范相对路径")
        source = Path(local_path)
        container = self._container(container_id)
        # 大型数据文件在磁盘暂存 tar，避免上传时把整个数据集读入内存。
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w") as bundle:
                for depth in range(1, len(parts)):
                    directory = tarfile.TarInfo("/".join(parts[:depth]))
                    directory.type = tarfile.DIRTYPE
                    directory.mode = 0o755
                    bundle.addfile(directory)
                info = bundle.gettarinfo(str(source), arcname=remote_relative_path)
                if not info.isfile():
                    raise ValueError("上传源必须是普通文件")
                info.mode, info.uid, info.gid, info.uname, info.gname = 0o644, 0, 0, "", ""
                with source.open("rb") as content:
                    bundle.addfile(info, content)
            archive.seek(0)
            accepted = _invoke(lambda: container.put_archive("/lfa", archive))
            if not accepted:
                raise GatewayError(f"Docker 未接受上传文件：{remote_relative_path}")

    @staticmethod
    def _parse_result(output: str) -> dict[str, Any]:
        for line in reversed(output.splitlines()):
            if line.startswith(RESULT_PREFIX):
                try:
                    result = json.loads(line[len(RESULT_PREFIX) :])
                except json.JSONDecodeError as error:
                    raise GatewayError("远端工具返回了无效 JSON") from error
                if not isinstance(result, dict):
                    raise GatewayError("远端工具结果必须为 JSON 对象")
                return result
        raise GatewayError(f"远端工具未返回结构化结果：\n{output[-8000:]}")

    def worker_result(self, container_id: str) -> dict[str, Any]:
        """解析后台工具最后一条结构化输出，返回结果对象。"""
        return self._parse_result(self.logs(container_id, tail=2000))

    def ensure_storage(self) -> None:
        """初始化远端受管根目录及模型、数据、缓存和任务子目录。"""
        self.run_worker("init", {})
