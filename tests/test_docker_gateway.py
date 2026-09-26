"""验证 Docker 参数和传输语义，不连接远端或启动真实容器。"""

import io
import json
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from docker.errors import APIError, DockerException, NotFound
from requests.exceptions import ConnectionError, ReadTimeout

from llamafactory_agent.config import Settings
from llamafactory_agent.docker_gateway import (
    ContainerMissing,
    DockerGateway,
    GatewayError,
    GatewayUnavailable,
    external_model_path,
)


@pytest.fixture
def gateway(monkeypatch):
    client = MagicMock()
    image = MagicMock(id="sha256:fixed-image", tags=["training:stable"])
    client.images.get.return_value = image
    client.version.return_value = {"Version": "28.0.0", "ApiVersion": "1.48"}
    container = MagicMock(id="container-id")
    container.put_archive.return_value = True
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b'initial log\nLFA_RESULT={"ready":true}\n'
    client.containers.create.return_value = container
    client.containers.get.return_value = container
    monkeypatch.setattr("llamafactory_agent.docker_gateway.docker.DockerClient", lambda **_: client)
    settings = Settings(_env_file=None, model_scan_dirs=["/opt/models"])
    return DockerGateway(settings)


def tar_bytes(content: bytes, name: str = "log.jsonl") -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    return archive.getvalue()


def test_training_binds_gpu_and_pins_legacy_offline_execution(gateway):
    assert gateway.create_training("training-123", "GPU-123") == "container-id"
    arguments = gateway.client.containers.create.call_args.kwargs
    assert arguments["image"] == "sha256:fixed-image"
    assert arguments["command"] == ["llamafactory-cli", "train", "/lfa/jobs/training-123/train.yaml"]
    assert arguments["device_requests"][0]["DeviceIDs"] == ["GPU-123"]
    assert arguments["device_requests"][0]["Capabilities"] == [["gpu"]]
    assert arguments["environment"]["USE_V1"] == "0"
    assert arguments["environment"]["USE_MODELSCOPE_HUB"] == "1"
    assert arguments["environment"]["HF_HUB_OFFLINE"] == "1"
    assert arguments["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert arguments["environment"]["HF_DATASETS_OFFLINE"] == "1"
    assert arguments["environment"]["HF_HOME"] == "/lfa/jobs/training-123/cache/huggingface"
    assert arguments["environment"]["HF_DATASETS_CACHE"] == "/lfa/jobs/training-123/cache/datasets"
    assert arguments["environment"]["MODELSCOPE_CACHE"] == "/lfa/jobs/training-123/cache/modelscope"
    assert arguments["restart_policy"] == {"Name": "no"}
    assert arguments["init"] is True
    assert arguments["shm_size"] == "2g"
    mounts = {mount["Target"]: mount for mount in arguments["mounts"]}
    assert mounts["/lfa"]["ReadOnly"] is True
    external_path = external_model_path("/opt/models")
    assert mounts[external_path]["ReadOnly"] is True
    assert mounts[external_path]["Source"] == "/opt/models"
    assert mounts["/lfa/jobs/training-123"]["ReadOnly"] is False
    assert mounts["/lfa/jobs/training-123"]["Source"].endswith("/jobs/training-123")
    gateway.client.containers.create.return_value.start.assert_not_called()


def test_worker_creation_uploads_program_without_starting(gateway):
    payload = {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "中文名称"}
    assert gateway.create_worker("download-1", "model_download", payload) == "container-id"
    arguments = gateway.client.containers.create.call_args.kwargs
    assert arguments["name"] == "lfa-download-1"
    assert arguments["labels"] == {"lfa.managed": "true", "lfa.job_id": "download-1", "lfa.worker": "true"}
    assert arguments["command"][:4] == ["python", "-B", "/tmp/lfa-worker.py", "model_download"]
    assert json.loads(arguments["command"][4]) == payload
    assert arguments["mounts"][0]["ReadOnly"] is False
    container = gateway.client.containers.create.return_value
    directory, archive = container.put_archive.call_args.args
    assert directory == "/tmp"
    archive.seek(0)
    with tarfile.open(fileobj=archive) as bundle:
        assert bundle.getnames() == ["lfa-worker.py"]
        assert b"RESULT_PREFIX" in bundle.extractfile("lfa-worker.py").read()
    container.start.assert_not_called()


def test_resource_probe_mounts_existing_parent_and_requests_gpu(gateway):
    assert gateway.run_worker("resources", {}, gpus=True) == {"ready": True}
    arguments = gateway.client.containers.create.call_args.kwargs
    assert arguments["mounts"][0]["Source"] == "/tmp"
    assert arguments["mounts"][0]["ReadOnly"] is True
    assert arguments["device_requests"][0]["Count"] == -1
    container = gateway.client.containers.create.return_value
    container.start.assert_called_once()
    container.remove.assert_called_once_with(force=True)


def test_catalog_does_not_mount_host_storage(gateway):
    gateway.run_worker("catalog", {})
    arguments = gateway.client.containers.create.call_args.kwargs
    assert "mounts" not in arguments
    assert "volumes" not in arguments


def test_dataset_catalog_only_mounts_external_sources(gateway):
    gateway.run_worker(
        "datasets_catalog", {}, extra_mounts={"/opt/LlamaFactory/data": "/external/datasets/0"}
    )
    mounts = gateway.client.containers.create.call_args.kwargs["mounts"]
    assert len(mounts) == 1
    assert mounts[0]["Source"] == "/opt/LlamaFactory/data"
    assert mounts[0]["Target"] == "/external/datasets/0"
    assert mounts[0]["ReadOnly"] is True


@pytest.mark.parametrize("action", ["scan_models", "preflight"])
def test_scan_and_preflight_mount_storage_and_explicit_sources_readonly(gateway, action):
    gateway.run_worker(action, {}, extra_mounts={"/opt/models": "/external/models/0"})
    arguments = gateway.client.containers.create.call_args.kwargs
    assert all(mount["ReadOnly"] for mount in arguments["mounts"])
    assert arguments["mounts"][1]["Source"] == "/opt/models"


def test_storage_initialization_alone_allows_bind_source_creation(gateway):
    gateway.ensure_storage()
    arguments = gateway.client.containers.create.call_args.kwargs
    assert arguments["volumes"] == {"/tmp/llamafactory-agent": {"bind": "/lfa", "mode": "rw"}}
    assert "mounts" not in arguments


def test_worker_failure_includes_stderr_and_cleans_container(gateway):
    container = gateway.client.containers.create.return_value
    container.wait.return_value = {"StatusCode": 1}
    container.logs.return_value = b"ValueError: missing weights"
    with pytest.raises(GatewayError, match="missing weights"):
        gateway.run_worker("scan_models", {})
    container.remove.assert_called_once_with(force=True)


def test_worker_timeout_remains_connectivity_error_and_cleans_container(gateway):
    container = gateway.client.containers.create.return_value
    container.wait.side_effect = ReadTimeout("timeout")
    with pytest.raises(GatewayUnavailable):
        gateway.run_worker("scan_models", {})
    container.remove.assert_called_once_with(force=True)


def test_failed_script_upload_cleans_unstarted_container(gateway):
    container = gateway.client.containers.create.return_value
    container.put_archive.side_effect = ConnectionError("lost connection")
    with pytest.raises(GatewayUnavailable):
        gateway.create_worker("download-1", "model_download", {})
    container.remove.assert_called_once_with(force=True)
    container.start.assert_not_called()


def test_recovered_created_worker_receives_program_before_start(gateway):
    container = gateway.client.containers.get.return_value
    container.attrs = {"Config": {"Labels": {"lfa.worker": "true"}}, "State": {"Status": "created"}}
    actions = []
    container.put_archive.side_effect = lambda *_: actions.append("upload") or True
    container.start.side_effect = lambda: actions.append("start")
    gateway.start("recovered-worker")
    assert actions == ["upload", "start"]


def test_recovery_does_not_start_worker_when_program_upload_fails(gateway):
    container = gateway.client.containers.get.return_value
    container.attrs = {"Config": {"Labels": {"lfa.worker": "true"}}, "State": {"Status": "created"}}
    container.put_archive.side_effect = ConnectionError("lost connection")
    with pytest.raises(GatewayUnavailable):
        gateway.start("recovered-worker")
    container.start.assert_not_called()


def test_start_training_does_not_upload_worker_program(gateway):
    container = gateway.client.containers.get.return_value
    container.attrs = {"Config": {"Labels": {"lfa.job_id": "training-123"}}, "State": {"Status": "created"}}
    gateway.start("training-container")
    container.put_archive.assert_not_called()
    container.start.assert_called_once()


def test_inspect_preserves_raw_container_state(gateway):
    container = gateway.client.containers.get.return_value
    container.attrs = {
        "State": {"Status": "exited", "Running": False, "ExitCode": 137, "OOMKilled": True, "Error": ""}
    }
    assert gateway.inspect("job-container") == {
        "status": "exited",
        "running": False,
        "exit_code": 137,
        "oom_killed": True,
        "error": "",
    }


@pytest.mark.parametrize(
    "exception", [ConnectionError("lost connection"), DockerException("api version failed")]
)
def test_connection_failures_are_distinct_from_missing_container(gateway, exception):
    gateway.client.containers.get.side_effect = exception
    with pytest.raises(GatewayUnavailable):
        gateway.inspect("job-container")


def test_missing_container_has_separate_exception(gateway):
    gateway.client.containers.get.side_effect = NotFound("container removed")
    with pytest.raises(ContainerMissing):
        gateway.inspect("job-container")


def test_daemon_validation_error_is_not_connection_failure(gateway):
    gateway.client.containers.get.side_effect = APIError("invalid device request")
    with pytest.raises(GatewayError) as failure:
        gateway.inspect("job-container")
    assert not isinstance(failure.value, GatewayUnavailable)


def test_find_job_uses_management_labels_and_includes_stopped_containers(gateway):
    gateway.client.containers.list.return_value = [MagicMock(id="found-id")]
    assert gateway.find_job("training-123") == "found-id"
    gateway.client.containers.list.assert_called_once_with(
        all=True, filters={"label": ["lfa.managed=true", "lfa.job_id=training-123"]}
    )


def test_read_file_uses_archive_without_starting_stopped_container(gateway):
    content = '{"loss": 0.25}\n'.encode()
    container = gateway.client.containers.get.return_value
    archive = tar_bytes(content)
    container.get_archive.return_value = (
        iter(archive[offset : offset + 23] for offset in range(0, len(archive), 23)),
        {},
    )
    assert gateway.read_file("stopped-container", "/lfa/jobs/job-1/trainer_log.jsonl") == content
    container.start.assert_not_called()


def test_read_file_rejects_large_files_before_collecting_contents(gateway):
    container = gateway.client.containers.get.return_value
    archive = tar_bytes(b"a" * 128)
    container.get_archive.return_value = (iter([archive]), {})
    with pytest.raises(GatewayError, match="超过读取上限"):
        gateway.read_file("stopped-container", "/lfa/large.bin", max_bytes=100)


def test_archive_errors_during_streaming_preserve_connection_semantics(gateway):
    def broken_stream():
        yield b"first chunk"
        raise ConnectionError("stream interrupted")

    gateway.client.containers.get.return_value.get_archive.return_value = (broken_stream(), {})
    response = gateway.archive("stopped-container", "/lfa/jobs/job-1/output")
    assert next(response) == b"first chunk"
    with pytest.raises(GatewayUnavailable):
        next(response)


def test_upload_to_stopped_container_includes_parents_without_start(tmp_path, gateway):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"instruction":"test"}\n')
    captured = {}

    def receive_archive(path, content):
        captured["path"] = path
        captured["content"] = content.read()
        return True

    container = gateway.client.containers.get.return_value
    container.put_archive.side_effect = receive_archive
    gateway.upload_file("stopped-container", source, "incoming/dataset-1/source.jsonl")
    assert captured["path"] == "/lfa"
    with tarfile.open(fileobj=io.BytesIO(captured["content"])) as bundle:
        assert bundle.getnames() == ["incoming", "incoming/dataset-1", "incoming/dataset-1/source.jsonl"]
        assert bundle.extractfile("incoming/dataset-1/source.jsonl").read() == source.read_bytes()
    container.start.assert_not_called()


@pytest.mark.parametrize(
    "path", ["/tmp/data", "../data", "jobs/../data", "jobs//data", "jobs/./data", "jobs\\data", ""]
)
def test_upload_rejects_noncanonical_paths(tmp_path, gateway, path):
    with pytest.raises(ValueError, match="规范相对路径"):
        gateway.upload_file("stopped-container", tmp_path / "source", path)


def test_worker_result_uses_last_result_line(gateway):
    container = gateway.client.containers.get.return_value
    container.logs.return_value = b'LFA_RESULT={"ready":false}\nother log\nLFA_RESULT={"ready":true}\n'
    assert gateway.worker_result("container-id") == {"ready": True}


@pytest.mark.parametrize("content", [b"no result", b"LFA_RESULT=invalid", b"LFA_RESULT=[]"])
def test_worker_result_requires_json_object(gateway, content):
    gateway.client.containers.get.return_value.logs.return_value = content
    with pytest.raises(GatewayError):
        gateway.worker_result("container-id")


def test_ping_reports_resolved_image_id(gateway):
    assert gateway.ping() == {
        "docker_version": "28.0.0",
        "api_version": "1.48",
        "image_id": "sha256:fixed-image",
        "image_tags": ["training:stable"],
    }


def test_gateway_creation_and_close_do_not_connect_while_remote_is_offline(monkeypatch):
    constructor = MagicMock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr("llamafactory_agent.docker_gateway.docker.DockerClient", constructor)
    gateway = DockerGateway(Settings(_env_file=None))
    gateway.close()
    constructor.assert_not_called()
    with pytest.raises(GatewayUnavailable):
        gateway.ping()


def test_external_mount_alias_is_stable_across_directory_order_and_spelling(gateway):
    gateway.settings.model_scan_dirs = ["/opt/models", "/mnt/models"]
    gateway.create_training("first", "GPU-123")
    first = {
        mount["Source"]: mount["Target"]
        for mount in gateway.client.containers.create.call_args.kwargs["mounts"]
        if mount["Target"].startswith("/external/models/")
    }
    gateway.settings.model_scan_dirs.reverse()
    gateway.create_training("second", "GPU-123")
    second = {
        mount["Source"]: mount["Target"]
        for mount in gateway.client.containers.create.call_args.kwargs["mounts"]
        if mount["Target"].startswith("/external/models/")
    }
    assert first == second
    assert first["/opt/models"] != first["/mnt/models"]
    assert external_model_path("/opt//models/") == external_model_path("/opt/models")


def job_container(gateway, *, status="exited", running=False):
    container = gateway.client.containers.get.return_value
    container.attrs = {
        "Config": {"Labels": {"lfa.managed": "true", "lfa.job_id": "job-1"}},
        "Mounts": [{"Type": "bind", "Source": gateway.settings.storage_root, "Destination": "/lfa"}],
        "State": {"Status": status, "Running": running, "ExitCode": 0},
    }
    return container


def reader_container(gateway, *, owner="previous-process", status="created", running=False):
    container = MagicMock(id="reader-id")
    container.attrs = {
        "Config": {
            "Labels": {
                "lfa.managed": "true",
                "lfa.artifact_reader": "true",
                "lfa.artifact_job_id": "job-1",
                "lfa.storage_root": gateway.settings.storage_root,
                "lfa.controller_scope": gateway._reader_scope,
                "lfa.owner": owner,
            }
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": f"{gateway.settings.storage_root}/jobs/job-1/output",
                "Destination": "/output",
                "RW": False,
            }
        ],
        "State": {"Status": status, "Running": running, "ExitCode": 0},
    }
    return container


def test_inspect_job_requires_matching_labels_and_storage(gateway):
    job_container(gateway)
    assert gateway.inspect_job("container-id", "job-1") == {
        "status": "exited",
        "running": False,
        "exit_code": 0,
        "oom_killed": False,
        "error": "",
    }


@pytest.mark.parametrize(
    "labels",
    [{}, {"lfa.managed": "false", "lfa.job_id": "job-1"}, {"lfa.managed": "true", "lfa.job_id": "other-job"}],
)
def test_remove_job_rejects_missing_or_mismatched_labels(gateway, labels):
    container = job_container(gateway)
    container.attrs["Config"]["Labels"] = labels
    with pytest.raises(GatewayError, match="标签"):
        gateway.remove_job("container-id", "job-1")
    container.remove.assert_not_called()


@pytest.mark.parametrize(
    "mounts",
    [
        [],
        [{"Type": "bind", "Source": "/other/project", "Destination": "/lfa"}],
        [{"Type": "volume", "Source": "/tmp/llamafactory-agent", "Destination": "/lfa"}],
    ],
)
def test_remove_job_rejects_unrelated_or_missing_storage(gateway, mounts):
    container = job_container(gateway)
    container.attrs["Mounts"] = mounts
    with pytest.raises(GatewayError, match="挂载"):
        gateway.remove_job("container-id", "job-1")
    container.remove.assert_not_called()


@pytest.mark.parametrize(
    "status,running", [("running", True), ("paused", True), ("restarting", False), ("exited", True)]
)
def test_remove_job_refuses_running_or_transitional_containers(gateway, status, running):
    container = job_container(gateway, status=status, running=running)
    with pytest.raises(GatewayError, match="尚未停止"):
        gateway.remove_job("container-id", "job-1")
    container.remove.assert_not_called()


@pytest.mark.parametrize("status", ["created", "exited", "dead"])
def test_remove_job_does_not_force_or_remove_volumes(gateway, status):
    container = job_container(gateway, status=status)
    gateway.remove_job("container-id", "job-1")
    container.remove.assert_called_once_with()


def test_stream_logs_reads_all_output_without_following_and_closes(gateway):
    container = job_container(gateway)
    chunks = MagicMock()
    chunks.__next__.side_effect = [b"stdout\n", b"stderr\n", StopIteration]
    container.logs.return_value = chunks
    assert list(gateway.stream_logs("container-id")) == [b"stdout\n", b"stderr\n"]
    container.logs.assert_called_once_with(stdout=True, stderr=True, stream=True, follow=False)
    chunks.close.assert_called_once()


def test_stream_logs_created_container_has_no_log_request(gateway):
    container = job_container(gateway, status="created")
    assert list(gateway.stream_logs("container-id")) == []
    container.logs.assert_not_called()


def test_logs_reports_missing_container_when_removed_after_lookup(gateway):
    container = job_container(gateway)
    container.logs.side_effect = NotFound("container removed between inspect and logs")
    with pytest.raises(ContainerMissing):
        gateway.logs("container-id")
    gateway.client.containers.get.assert_called_once_with("container-id")
    container.logs.assert_called_once_with(stdout=True, stderr=True, tail=200)


@pytest.mark.parametrize("abort", ["close", "error"])
def test_stream_logs_closes_on_disconnect_or_remote_error(gateway, abort):
    container = job_container(gateway)
    chunks = MagicMock()
    chunks.__next__.side_effect = [b"partial\n", ConnectionError("stream failed")]
    container.logs.return_value = chunks
    stream = gateway.stream_logs("container-id")
    assert next(stream) == b"partial\n"
    if abort == "close":
        stream.close()
    else:
        with pytest.raises(GatewayUnavailable):
            next(stream)
    chunks.close.assert_called_once()


def test_artifact_archive_uses_unstarted_readonly_reader_and_cleans_it(gateway):
    container = gateway.client.containers.create.return_value
    chunks = MagicMock()
    chunks.__next__.side_effect = [b"tar-content", StopIteration]
    container.get_archive.return_value = (chunks, {})
    assert list(gateway.artifact_archive("job-1", "checkpoint-3/model.safetensors")) == [b"tar-content"]
    arguments = gateway.client.containers.create.call_args.kwargs
    assert arguments["mounts"][0]["Source"] == f"{gateway.settings.storage_root}/jobs/job-1/output"
    assert arguments["mounts"][0]["Target"] == "/output"
    assert arguments["mounts"][0]["ReadOnly"] is True
    assert arguments["labels"]["lfa.artifact_reader"] == "true"
    assert arguments["labels"]["lfa.owner"] == gateway._reader_owner
    assert arguments["labels"]["lfa.controller_scope"] == gateway._reader_scope
    assert "lfa.job_id" not in arguments["labels"]
    container.get_archive.assert_called_once_with("/output/checkpoint-3/model.safetensors", chunk_size=65536)
    container.start.assert_not_called()
    container.remove.assert_called_once_with()
    gateway.client.containers.get.assert_not_called()
    chunks.close.assert_called_once()
    assert not gateway._active_readers


@pytest.mark.parametrize("abort", ["close", "error"])
def test_artifact_archive_cleans_reader_after_disconnect_or_stream_error(gateway, abort):
    container = gateway.client.containers.create.return_value
    chunks = MagicMock()
    chunks.__next__.side_effect = [b"partial tar", ConnectionError("stream failed")]
    container.get_archive.return_value = (chunks, {})
    stream = gateway.artifact_archive("job-1", "model.safetensors")
    assert next(stream) == b"partial tar"
    assert container.id in gateway._active_readers
    if abort == "close":
        stream.close()
    else:
        with pytest.raises(GatewayUnavailable):
            next(stream)
    chunks.close.assert_called_once()
    container.remove.assert_called_once_with()
    assert not gateway._active_readers


def test_artifact_archive_cleans_reader_when_archive_creation_fails(gateway):
    container = gateway.client.containers.create.return_value
    container.get_archive.side_effect = NotFound("artifact removed")
    with pytest.raises(GatewayError):
        list(gateway.artifact_archive("job-1", "model.safetensors"))
    container.remove.assert_called_once_with()


@pytest.mark.parametrize(
    "job_id,path",
    [
        ("../job", "model.bin"),
        ("job-1", "../model.bin"),
        ("job-1", "/model.bin"),
        ("job-1", "output\\file.bin"),
    ],
)
def test_artifact_archive_rejects_path_escape_before_creating_container(gateway, job_id, path):
    with pytest.raises(ValueError):
        list(gateway.artifact_archive(job_id, path))
    gateway.client.containers.create.assert_not_called()


def test_artifact_reader_cleanup_matches_previous_controller_and_storage(gateway):
    container = reader_container(gateway)
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    assert gateway.cleanup_artifact_readers() == 1
    container.remove.assert_called_once_with()
    labels = gateway.client.containers.list.call_args.kwargs["filters"]["label"]
    assert f"lfa.controller_scope={gateway._reader_scope}" in labels


@pytest.mark.parametrize(
    "label,value",
    [
        ("lfa.managed", "false"),
        ("lfa.artifact_reader", "false"),
        ("lfa.controller_scope", "another-controller"),
        ("lfa.storage_root", "/tmp/another-project"),
        ("lfa.owner", ""),
    ],
)
def test_artifact_reader_cleanup_does_not_trust_list_filters_alone(gateway, label, value):
    container = reader_container(gateway)
    container.attrs["Config"]["Labels"][label] = value
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    assert gateway.cleanup_artifact_readers() == 0
    container.remove.assert_not_called()


@pytest.mark.parametrize("change", ["source", "writable", "running", "active"])
def test_artifact_reader_cleanup_rejects_unowned_or_active_containers(gateway, change):
    container = reader_container(gateway)
    if change == "source":
        container.attrs["Mounts"][0]["Source"] = "/tmp/another-project/output"
    elif change == "writable":
        container.attrs["Mounts"][0]["RW"] = True
    elif change == "running":
        container.attrs["State"].update(Status="running", Running=True)
    else:
        gateway._active_readers.add(container.id)
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    assert gateway.cleanup_artifact_readers() == 0
    container.remove.assert_not_called()


def test_failed_reader_removal_is_retried_without_interrupting_completed_download(gateway):
    container = reader_container(gateway, owner=gateway._reader_owner)
    gateway.client.containers.create.return_value = container
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    container.get_archive.return_value = (iter([b"complete tar"]), {})
    container.remove.side_effect = [ConnectionError("offline"), None]
    assert list(gateway.artifact_archive("job-1", "model.bin")) == [b"complete tar"]
    assert not gateway._active_readers
    assert gateway.cleanup_artifact_readers() == 1
    assert container.remove.call_count == 2


def test_reader_cleanup_network_failure_can_be_retried(gateway):
    container = reader_container(gateway)
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    container.remove.side_effect = [ConnectionError("offline"), None]
    with pytest.raises(GatewayUnavailable):
        gateway.cleanup_artifact_readers()
    assert gateway.cleanup_artifact_readers() == 1


def test_reader_scope_distinguishes_projects_sharing_the_same_remote_root(gateway, tmp_path):
    other_settings = Settings(_env_file=None, runtime_dir=tmp_path / "other-controller")
    other_gateway = DockerGateway(other_settings)
    assert other_gateway._reader_scope != gateway._reader_scope


def test_reader_with_lost_create_response_is_recovered_in_same_process(gateway):
    container = reader_container(gateway, owner=gateway._reader_owner)
    gateway.client.containers.create.side_effect = ConnectionError("created but response lost")
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.return_value = container
    with pytest.raises(GatewayUnavailable):
        list(gateway.artifact_archive("job-1", "model.bin"))
    assert not gateway._active_readers
    assert gateway.cleanup_artifact_readers() == 1
    container.remove.assert_called_once_with()


def test_reader_cleanup_waits_for_create_and_active_registration(gateway):
    container = reader_container(gateway, owner=gateway._reader_owner)
    creation_started = threading.Event()
    allow_creation = threading.Event()
    cleanup_inspected = threading.Event()

    def delayed_create(**_):
        creation_started.set()
        assert allow_creation.wait(timeout=5)
        return container

    def inspect_reader(_):
        cleanup_inspected.set()
        return container

    gateway.client.containers.create.side_effect = delayed_create
    gateway.client.containers.list.return_value = [container]
    gateway.client.containers.get.side_effect = inspect_reader
    container.get_archive.return_value = (iter([b"tar chunk"]), {})
    stream = gateway.artifact_archive("job-1", "model.bin")
    with ThreadPoolExecutor(max_workers=2) as executor:
        download = executor.submit(next, stream)
        assert creation_started.wait(timeout=5)
        cleanup = executor.submit(gateway.cleanup_artifact_readers)
        assert cleanup_inspected.wait(timeout=5)
        assert not cleanup.done()
        allow_creation.set()
        assert download.result(timeout=5) == b"tar chunk"
        assert cleanup.result(timeout=5) == 0
    container.remove.assert_not_called()
    stream.close()
    container.remove.assert_called_once_with()
