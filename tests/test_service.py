"""验证调度恢复和资源边界，避免重复启动实际 GPU 任务。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llamafactory_agent.config import Settings
from llamafactory_agent.docker_gateway import FileTooLarge, GatewayUnavailable
from llamafactory_agent.service import ServiceError, TrainingService, safe_relative_path


@pytest.fixture
def service(tmp_path):
    settings = Settings(_env_file=None, runtime_dir=tmp_path, docker_host="tcp://example.invalid:2375")
    gateway = MagicMock()
    gateway.find_job.return_value = None
    gateway.create_worker.return_value = "worker-container"
    gateway.inspect.return_value = {"status": "created", "running": False, "exit_code": 0}
    instance = TrainingService(settings, gateway)
    instance.store.initialize()
    return instance


def seed_resources(service):
    service.store.save_resource("model", "model1", "1", {"path": "/lfa/models/model1", "complete": True})
    service.store.save_resource(
        "dataset",
        "dataset1",
        "version1",
        {
            "path": "/lfa/datasets/dataset1/version1/data.jsonl",
            "ranking": False,
        },
    )


def test_config_accepts_injected_docker_host(tmp_path):
    settings = Settings(_env_file=None, docker_host="tcp://example.invalid:2375", runtime_dir=tmp_path)
    assert settings.docker_host == "tcp://example.invalid:2375"


def test_idempotency_reuses_job_and_rejects_different_payload(service):
    first = service.download_model("Qwen/test", idempotency_key="same")
    repeated = service.download_model("Qwen/test", idempotency_key="same")
    assert repeated["id"] == first["id"]
    with pytest.raises(ServiceError) as error:
        service.download_model("Qwen/other", idempotency_key="same")
    assert error.value.status_code == 409
    assert len(service.store.jobs()) == 1


def test_reconnect_finds_container_without_creating_duplicate(service):
    job = service.download_model("Qwen/test")
    service.gateway.create_worker.side_effect = GatewayUnavailable("创建响应丢失")
    service.tick()
    assert service.store.job(job["id"])["status"] == "starting"
    service.gateway.find_job.return_value = "existing-container"
    service.gateway.create_worker.side_effect = None
    service.tick()
    assert service.store.job(job["id"])["container_id"] == "existing-container"
    assert service.gateway.create_worker.call_count == 1
    service.gateway.start.assert_called_once_with("existing-container")


def test_running_connection_loss_is_not_failure(service):
    job = service.download_model("Qwen/test")
    service.store.update_job(job["id"], status="running", container_id="running-container")
    service.gateway.inspect.side_effect = GatewayUnavailable("网络断开")
    service.tick()
    assert service.store.job(job["id"])["status"] == "running"
    service.gateway.create_worker.assert_not_called()


def test_start_response_lost_does_not_start_running_container_twice(service):
    job = service.download_model("Qwen/test")
    service.gateway.start.side_effect = GatewayUnavailable("启动响应丢失")
    service.tick()
    assert service.store.job(job["id"])["status"] == "starting"
    service.gateway.inspect.return_value = {"status": "running", "running": True, "exit_code": 0}
    service.tick()
    assert service.store.job(job["id"])["status"] == "running"
    service.gateway.start.assert_called_once()


def test_cancel_queued_job_never_creates_container(service):
    job = service.download_model("Qwen/test")
    assert service.cancel(job["id"])["status"] == "cancelled"
    service.tick()
    service.gateway.create_worker.assert_not_called()


def test_one_transfer_job_at_a_time(service):
    first = service.download_model("Qwen/first")
    second = service.download_model("Qwen/second")
    service.tick()
    assert service.store.job(first["id"])["status"] == "running"
    assert service.store.job(second["id"])["status"] == "queued"
    service.gateway.create_worker.assert_called_once()


def test_starting_training_without_gpu_fails_instead_of_sticking(service):
    seed_resources(service)
    job = service.create_training("model1", "dataset1", training_args={"template": "qwen"})
    service.store.update_job(job["id"], status="starting")
    service.gateway.run_worker.return_value = {"gpus": []}
    service.tick()
    assert service.store.job(job["id"])["status"] == "failed"


def test_training_freezes_dataset_version_and_disallows_remote_overrides(service):
    seed_resources(service)
    job = service.create_training("model1", "dataset1", training_args={"template": "qwen"})
    service.store.save_resource("dataset", "dataset1", "version2", {"path": "/new", "ranking": False})
    assert job["payload"]["dataset"]["version"] == "version1"
    with pytest.raises(ServiceError, match="由服务管理"):
        service.create_training(
            "model1", "dataset1", training_args={"template": "qwen", "dataset": "hub/name"}
        )


@pytest.mark.parametrize("path", ["/etc/passwd", "../config.json", "nested/../../config", "folder\\file"])
def test_artifact_path_cannot_escape(path):
    with pytest.raises(ServiceError):
        safe_relative_path(path)


def test_remote_dataset_source_must_be_in_registered_directory(service):
    with pytest.raises(ServiceError, match="源文件不在"):
        service.import_dataset(source_path="/etc/config.json")
    with pytest.raises(ServiceError):
        service.import_dataset(source_path="https://modelscope.cn/datasets/example")


def test_uploaded_file_deduplication_uses_content_not_temporary_name(service):
    uploads = service.settings.runtime_dir / "uploads"
    uploads.mkdir()
    first_path, second_path = uploads / "first.jsonl", uploads / "second.jsonl"
    first_path.write_text('{"instruction":"问","output":"答"}\n', encoding="utf-8")
    second_path.write_bytes(first_path.read_bytes())
    first = service.import_dataset(local_path=first_path, filename="sample.jsonl", idempotency_key="upload")
    repeated = service.import_dataset(
        local_path=second_path, filename="sample.jsonl", idempotency_key="upload"
    )
    assert first["id"] == repeated["id"]
    assert first_path.exists()
    assert not second_path.exists()


def test_restore_local_upload_after_created_container(service):
    uploads = service.settings.runtime_dir / "uploads"
    uploads.mkdir()
    upload = uploads / "data.jsonl"
    upload.write_text('{"instruction":"问","output":"答"}\n', encoding="utf-8")
    job = service.import_dataset(local_path=upload, filename="data.jsonl")
    service.store.update_job(job["id"], status="starting")
    service.gateway.find_job.return_value = "existing-container"
    service.tick()
    service.gateway.upload_file.assert_called_once_with(
        "existing-container", Path(upload), f"incoming/{job['id']}/upload.jsonl"
    )
    service.gateway.start.assert_called_once_with("existing-container")


def test_nonzero_exit_archives_logs_and_does_not_register_resource(service):
    job = service.download_model("Qwen/test")
    service.store.update_job(job["id"], status="running", container_id="failed-container")
    service.gateway.inspect.return_value = {"status": "exited", "running": False, "exit_code": 1}
    service.gateway.inspect_job.return_value = service.gateway.inspect.return_value
    service.gateway.logs.return_value = "模型下载失败"
    service.gateway.stream_logs.return_value = iter(["模型下载失败\n".encode()])
    service.tick()
    assert service.store.job(job["id"])["status"] == "failed"
    assert service.store.resources("model") == []
    service.gateway.remove_job.assert_called_once_with("failed-container", job["id"])
    assert service.logs(job["id"])["logs"] == "模型下载失败\n"


def test_service_restarts_using_saved_job_and_container(service):
    job = service.download_model("Qwen/test")
    service.store.update_job(job["id"], status="running", container_id="saved-container")
    restarted = TrainingService(service.settings, service.gateway)
    service.gateway.inspect.return_value = {"status": "running", "running": True, "exit_code": 0}
    restarted.tick()
    assert restarted.store.job(job["id"])["container_id"] == "saved-container"
    service.gateway.create_worker.assert_not_called()


def test_large_metrics_log_uses_bounded_remote_tail(service):
    seed_resources(service)
    job = service.create_training("model1", "dataset1", training_args={"template": "qwen"})
    service.store.update_job(job["id"], status="running", container_id="training-container")
    service.gateway.read_file.side_effect = FileTooLarge("日志大于直接读取上限")
    service.gateway.run_worker.return_value = {"metrics": {"current_steps": 9000}}
    assert service.job(job["id"])["metrics"]["current_steps"] == 9000
    service.gateway.run_worker.assert_called_once_with("read_metrics", {"job_id": job["id"]})


def test_preflight_checks_actual_files_instead_of_only_saved_ready_flag(service):
    seed_resources(service)
    service.gateway.run_worker.side_effect = [
        {"valid": False, "errors": ["模型权重已被移除"]},
        {"gpus": [{"uuid": "gpu1"}]},
    ]
    service.gateway.ping.return_value = {}
    result = service.preflight("model1", "dataset1", training_args={"template": "qwen"})
    assert result["valid"] is False
    assert result["errors"] == ["模型权重已被移除"]


def test_controller_lock_prevents_duplicate_scheduler_and_releases_across_threads(service):
    competing = TrainingService(service.settings, service.gateway)
    service.start()
    try:
        with pytest.raises(RuntimeError, match="已有控制端"):
            competing.start()
    finally:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(service.close).result(timeout=5)
    competing.start()
    competing.close()
