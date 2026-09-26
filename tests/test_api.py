"""验证 HTTP 边界、来源限制及后台任务协议。"""

from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import anyio
import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from llamafactory_agent.api import ArtifactStreamingResponse, create_app
from llamafactory_agent.docker_gateway import GatewayError
from llamafactory_agent.service import ServiceError


class FakeService:
    """记录 API 调用，不连接远端 Docker。"""

    def __init__(self):
        self.calls = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def download_model(self, **kwargs):
        self.calls.append(("download_model", kwargs))
        return {"job_id": "download-job", "status": "queued"}

    def import_dataset(self, **kwargs):
        self.calls.append(("import_dataset", kwargs))
        return {"job_id": "import-job", "status": "queued"}

    def create_training(self, **kwargs):
        self.calls.append(("create_training", kwargs))
        return {"job_id": "training-job", "status": "queued"}

    def resume(self, job_id, checkpoint, idempotency_key=None):
        self.calls.append(("resume", (job_id, checkpoint, idempotency_key)))
        return {"job_id": "resume-job", "status": "queued"}

    def artifact_archive(self, job_id, relative_path):
        self.calls.append(("artifact_archive", (job_id, relative_path)))
        yield b"tar-"
        yield b"archive"

    def resources(self):
        raise GatewayError("Docker 暂时不可达")

    def job(self, job_id):
        raise KeyError(job_id)

    def cancel(self, job_id):
        raise ServiceError("任务已完成", status_code=409)


@pytest.fixture
def api_client(tmp_path):
    service = FakeService()
    settings = SimpleNamespace(runtime_dir=tmp_path, max_upload_bytes=1024)
    with TestClient(create_app(settings, service)) as client:
        yield client, service, tmp_path


def test_lifespan_and_local_health(tmp_path):
    service = FakeService()
    settings = SimpleNamespace(runtime_dir=tmp_path, max_upload_bytes=1024)
    with TestClient(create_app(settings, service)) as client:
        assert service.started
        assert client.get("/health").json() == {"status": "ok"}
        assert not service.closed
    assert service.closed


def test_model_download_forwards_idempotency_key(api_client):
    client, service, _ = api_client
    response = client.post(
        "/models/downloads",
        json={"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "master"},
        headers={"Idempotency-Key": "download-once"},
    )
    assert response.status_code == 202
    assert response.json()["job_id"] == "download-job"
    assert service.calls == [
        (
            "download_model",
            {
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "master",
                "idempotency_key": "download-once",
            },
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"repo_id": "https://huggingface.co/Qwen/model"},
        {"repo_id": "Qwen/model", "source": "huggingface"},
        {"repo_id": "Qwen/model", "token": "unrecognized"},
        {"repo_id": "model-without-namespace"},
    ],
)
def test_model_download_rejects_urls_and_unknown_fields(api_client, payload):
    client, service, _ = api_client
    assert client.post("/models/downloads", json=payload).status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"source_path": "https://modelscope.cn/datasets/example/file.json"},
        {"source_path": "relative/file.json"},
        {"source_path": "/tmp/../data/file.json"},
        {"source_path": "/tmp/data.csv"},
        {"source_path": "/tmp/data.json", "hub_url": "org/dataset"},
        {"source_path": "/tmp/data.json", "source": "modelscope"},
        {"source_path": "/tmp/data.json", "formatting": "unknown"},
    ],
)
def test_dataset_import_accepts_only_remote_text_files(api_client, payload):
    client, service, _ = api_client
    assert client.post("/datasets/imports", json=payload).status_code == 422
    assert service.calls == []


def test_remote_dataset_registration(api_client):
    client, service, _ = api_client
    response = client.post(
        "/datasets/imports",
        json={
            "source_path": "/opt/LlamaFactory/data/example.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversation"},
            "tags": {"role_tag": "role"},
            "ranking": True,
        },
        headers={"Idempotency-Key": "import-once"},
    )
    assert response.status_code == 202
    arguments = service.calls[0][1]
    assert arguments["local_path"] is None
    assert arguments["source_path"] == "/opt/LlamaFactory/data/example.json"
    assert arguments["columns"] == {"messages": "conversation"}
    assert arguments["ranking"] is True
    assert arguments["idempotency_key"] == "import-once"


def test_upload_uses_generated_local_filename(api_client):
    client, service, runtime_dir = api_client
    content = b'[{"instruction":"question","output":"answer"}]'
    response = client.post(
        "/datasets/uploads",
        files={"file": ("../../example.json", content, "application/json")},
        data={"columns": '{"prompt":"instruction"}'},
        headers={"Idempotency-Key": "upload-once"},
    )
    assert response.status_code == 202
    arguments = service.calls[0][1]
    local_path = Path(arguments["local_path"])
    assert local_path.parent == runtime_dir / "uploads"
    assert local_path.name != "example.json"
    assert local_path.read_bytes() == content
    assert arguments["source_path"] is None
    assert arguments["filename"] == "example.json"
    assert arguments["columns"] == {"prompt": "instruction"}
    assert arguments["idempotency_key"] == "upload-once"


def test_upload_size_limit_removes_partial_file(api_client):
    client, service, runtime_dir = api_client
    response = client.post("/datasets/uploads", files={"file": ("data.json", b"x" * 1025)})
    assert response.status_code == 413
    assert service.calls == []
    assert list((runtime_dir / "uploads").iterdir()) == []


@pytest.mark.parametrize("columns", ["[]", "invalid-json", '{"prompt":42}'])
def test_upload_rejects_invalid_mapping(api_client, columns):
    client, service, _ = api_client
    response = client.post(
        "/datasets/uploads", files={"file": ("data.json", b"[]")}, data={"columns": columns}
    )
    assert response.status_code == 422
    assert service.calls == []


def test_training_submission_forwards_pinned_version_and_idempotency(api_client):
    client, service, _ = api_client
    payload = {
        "model_id": "model-1",
        "dataset_id": "dataset-1",
        "dataset_version": "version-1",
        "training_args": {"stage": "sft", "finetuning_type": "lora", "max_steps": 3},
    }
    response = client.post("/training/jobs", json=payload, headers={"Idempotency-Key": "train-once"})
    assert response.status_code == 202
    assert service.calls == [("create_training", {**payload, "idempotency_key": "train-once"})]


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "..\\secret", "C:/secret", "x/../../secret"])
def test_artifact_download_rejects_path_traversal(api_client, path):
    client, service, _ = api_client
    response = client.get("/training/jobs/task/artifacts/download", params={"path": path})
    assert response.status_code == 400
    assert service.calls == []


def test_artifact_download_streams_tar(api_client):
    client, service, _ = api_client
    response = client.get("/training/jobs/task/artifacts/download", params={"path": "checkpoint-2"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-tar"
    assert response.content == b"tar-archive"
    assert service.calls == [("artifact_archive", ("task", "checkpoint-2"))]


def test_resume_rejects_external_checkpoint(api_client):
    client, service, _ = api_client
    response = client.post("/training/jobs/task/resume", json={"checkpoint": "../other-task/checkpoint-1"})
    assert response.status_code == 422
    assert service.calls == []


def test_resume_creates_new_task(api_client):
    client, service, _ = api_client
    response = client.post(
        "/training/jobs/task/resume",
        json={"checkpoint": "checkpoint-2"},
        headers={"Idempotency-Key": "resume-once"},
    )
    assert response.status_code == 202
    assert service.calls == [("resume", ("task", "checkpoint-2", "resume-once"))]


def test_error_statuses_distinguish_remote_missing_and_conflict(api_client):
    client, _, _ = api_client
    assert client.get("/resources").status_code == 503
    assert client.get("/jobs/missing").status_code == 404
    assert client.post("/jobs/completed/cancel").status_code == 409


def test_query_limits_and_scope_are_validated(api_client):
    client, service, _ = api_client
    assert client.get("/jobs/task/logs?tail=10001").status_code == 422
    assert client.get("/datasets/task/preview?limit=0").status_code == 422
    assert client.get("/datasets/task/preview?limit=21").status_code == 422
    assert client.get("/models?scope=huggingface").status_code == 422
    assert client.get("/training/jobs/task/artifacts/download").status_code == 422
    assert service.calls == []


class ClosingArchive:
    """记录原始归档的关闭次数，防止测试结果依赖垃圾回收。"""

    def __init__(self):
        self.parts = iter([b"tar-", b"archive"])
        self.close_count = 0
        self.closed_on_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.parts)

    def close(self):
        self.close_count += 1
        self.closed_on_thread = get_ident()


def test_archive_response_closes_source_after_normal_stream():
    source = ClosingArchive()
    response = ArtifactStreamingResponse(source, next(source))
    messages = []

    async def receive():
        raise AssertionError("ASGI 2.4 不应监听断开事件")

    async def send(message):
        messages.append(message)

    async def run_response():
        loop_thread = get_ident()
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert source.close_count == 1
        assert source.closed_on_thread != loop_thread

    anyio.run(run_response)
    assert b"".join(message.get("body", b"") for message in messages) == b"tar-archive"


@pytest.mark.parametrize("failure_event", ["http.response.start", "http.response.body"])
def test_archive_response_closes_source_on_asgi24_send_error(failure_event):
    source = ClosingArchive()
    response = ArtifactStreamingResponse(source, next(source))

    async def receive():
        raise AssertionError("ASGI 2.4 不应监听断开事件")

    async def send(message):
        if message["type"] == failure_event:
            raise OSError("客户端连接已断开")

    async def run_response():
        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert source.close_count == 1

    anyio.run(run_response)


def test_archive_response_closes_source_on_asgi20_disconnect():
    source = ClosingArchive()
    response = ArtifactStreamingResponse(source, next(source))

    async def run_response():
        body_started = anyio.Event()

        async def receive():
            await body_started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                body_started.set()
                await anyio.sleep_forever()

        with anyio.fail_after(5):
            await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        assert source.close_count == 1

    anyio.run(run_response)


def test_archive_response_shields_cleanup_from_caller_cancellation():
    source = ClosingArchive()
    response = ArtifactStreamingResponse(source, next(source))

    async def receive():
        raise AssertionError("ASGI 2.4 不应监听断开事件")

    async def run_response():
        async def send(message):
            scope.cancel()
            await anyio.sleep(0)

        with anyio.CancelScope() as scope:
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert source.close_count == 1

    anyio.run(run_response)
