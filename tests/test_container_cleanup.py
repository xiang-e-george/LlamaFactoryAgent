"""验证容器回收前的归档顺序、故障重试及历史任务可读性。"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from llamafactory_agent.config import Settings
from llamafactory_agent.docker_gateway import ContainerMissing, GatewayUnavailable
from llamafactory_agent.service import TrainingService
from llamafactory_agent.store import Store, timestamp


@pytest.fixture
def cleanup_service(tmp_path):
    settings = Settings(
        _env_file=None,
        runtime_dir=tmp_path,
        docker_host="tcp://example.invalid:2375",
        failed_container_retention_hours=0,
    )
    gateway = MagicMock()
    gateway.inspect_job.return_value = {
        "status": "exited",
        "running": False,
        "exit_code": 0,
        "oom_killed": False,
    }
    gateway.stream_logs.side_effect = lambda _: iter([b"first line\n", b"second line\n"])
    gateway.run_worker.return_value = {"metrics": {"current_steps": 3, "loss": 0.25}}
    service = TrainingService(settings, gateway)
    service.store.initialize()
    return service


def finished_job(
    service, job_id="training-job", *, status="succeeded", kind="training", container_id="container-1"
):
    """创建已终结任务，保留真实 SQLite 持久化行为。

    Args:
        service: 待验证的任务服务。
        job_id: 任务标识。
        status: 终结状态。
        kind: 任务类型。
        container_id: 可选的远端容器标识。
    Returns:
        持久化后的完整任务对象。
    """
    service.store.create_job(job_id, kind, {}, None, f"digest-{job_id}")
    return service.store.update_job(job_id, status=status, container_id=container_id)


def console_path(service, job_id):
    """返回约定的本地归档日志路径。

    Args:
        service: 待验证的任务服务。
        job_id: 任务标识。
    Returns:
        本地控制端日志文件路径。
    """
    return service.settings.runtime_dir / "jobs" / job_id / "console.log"


def test_archive_and_metadata_are_persisted_before_container_removal(cleanup_service):
    service = cleanup_service
    job = finished_job(service)

    def verify_removal(container_id, job_id):
        assert (container_id, job_id) == ("container-1", job["id"])
        assert console_path(service, job_id).read_bytes() == b"first line\nsecond line\n"
        persisted = service.store.job(job_id)
        assert persisted["archived_at"]
        assert persisted["exit_code"] == 0
        assert persisted["metrics"] == {"current_steps": 3, "loss": 0.25}
        assert persisted["container_removed_at"] is None

    service.gateway.remove_job.side_effect = verify_removal
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert persisted["status"] == "succeeded"
    assert persisted["container_removed_at"]
    assert persisted["cleanup_error"] is None
    service.gateway.inspect_job.assert_called_once_with("container-1", job["id"])
    service.gateway.remove_job.assert_called_once_with("container-1", job["id"])
    service.gateway.run_worker.assert_called_once_with("read_metrics", {"job_id": job["id"]})


def test_log_archive_failure_keeps_container_and_previous_complete_file(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    target = console_path(service, job["id"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous complete archive\n")

    def broken_log_stream(_):
        yield b"new partial content"
        raise GatewayUnavailable("日志传输中断")

    service.gateway.stream_logs.side_effect = broken_log_stream
    before = datetime.now(timezone.utc)
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert target.read_bytes() == b"previous complete archive\n"
    assert persisted["status"] == "succeeded"
    assert persisted["archived_at"] is None
    assert persisted["container_removed_at"] is None
    assert "日志传输中断" in persisted["cleanup_error"]
    retry = datetime.fromisoformat(persisted["cleanup_next_attempt_at"])
    assert before + timedelta(seconds=25) <= retry <= datetime.now(timezone.utc) + timedelta(seconds=35)
    service.gateway.remove_job.assert_not_called()


def test_metrics_archive_failure_keeps_container(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.gateway.run_worker.side_effect = GatewayUnavailable("指标归档暂时不可读")
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert persisted["status"] == "succeeded"
    assert not persisted["archived_at"]
    assert not persisted["container_removed_at"]
    assert persisted["cleanup_error"]
    service.gateway.remove_job.assert_not_called()


def test_delete_failure_retries_after_restart_without_archiving_twice(cleanup_service):
    service = cleanup_service
    job = finished_job(service, status="failed")
    service.store.update_job(job["id"], error="原始训练错误")
    service.gateway.inspect_job.return_value["exit_code"] = 2
    service.gateway.remove_job.side_effect = GatewayUnavailable("删除连接中断")
    service.cleanup_finished()
    first_attempt = service.store.job(job["id"])
    assert first_attempt["status"] == "failed"
    assert first_attempt["error"] == "原始训练错误"
    assert first_attempt["archived_at"]
    assert first_attempt["exit_code"] == 2
    assert not first_attempt["container_removed_at"]
    assert "删除连接中断" in first_attempt["cleanup_error"]
    assert first_attempt["cleanup_next_attempt_at"]

    restarted = TrainingService(service.settings, service.gateway)
    restarted.store.initialize()
    restarted.cleanup_finished()
    assert service.gateway.remove_job.call_count == 1
    restarted.store.update_job(job["id"], cleanup_next_attempt_at=None)
    service.gateway.remove_job.side_effect = None
    restarted.cleanup_finished()
    final = restarted.store.job(job["id"])
    assert final["status"] == "failed"
    assert final["error"] == "原始训练错误"
    assert final["archived_at"] == first_attempt["archived_at"]
    assert final["container_removed_at"]
    assert final["cleanup_error"] is None
    assert final["cleanup_next_attempt_at"] is None
    assert service.gateway.remove_job.call_count == 2
    service.gateway.stream_logs.assert_called_once()
    service.gateway.run_worker.assert_called_once()


def test_lost_remove_response_is_reconciled_from_missing_archived_container(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.gateway.remove_job.side_effect = GatewayUnavailable("删除响应丢失")
    service.cleanup_finished()
    archived_at = service.store.job(job["id"])["archived_at"]
    assert archived_at
    service.store.update_job(job["id"], cleanup_next_attempt_at=None)
    service.gateway.remove_job.side_effect = ContainerMissing("容器已经删除")
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert persisted["container_removed_at"]
    assert persisted["archived_at"] == archived_at
    assert persisted["cleanup_error"] is None
    assert persisted["status"] == "succeeded"
    assert service.gateway.remove_job.call_count == 2
    service.gateway.stream_logs.assert_called_once()


def test_unarchived_missing_container_records_loss_without_retrying_forever(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.gateway.inspect_job.side_effect = ContainerMissing("未归档容器已不存在")
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert persisted["status"] == "succeeded"
    assert not persisted["archived_at"]
    assert persisted["container_removed_at"]
    assert "容器在归档完成前已被移除，无法补齐归档" in persisted["cleanup_error"]
    assert persisted["cleanup_next_attempt_at"] is None
    service.gateway.stream_logs.assert_not_called()
    service.gateway.remove_job.assert_not_called()
    service.gateway.reset_mock()
    service.cleanup_finished()
    service.gateway.inspect_job.assert_not_called()


def test_finished_timestamp_is_set_once_and_survives_cleanup_changes(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    assert job["finished_at"]
    service.store.update_job(job["id"], cleanup_error="稍后重试")
    service.store.update_job(job["id"], status="succeeded")
    assert service.store.job(job["id"])["finished_at"] == job["finished_at"]


def test_failed_container_retention_uses_finished_time(cleanup_service):
    service = cleanup_service
    service.settings.failed_container_retention_hours = 24
    job = finished_job(service, status="failed")
    service.cleanup_finished()
    service.gateway.remove_job.assert_not_called()
    assert datetime.fromisoformat(service.store.job(job["id"])["cleanup_next_attempt_at"]) == (
        datetime.fromisoformat(job["finished_at"]) + timedelta(hours=24)
    )
    service.store.update_job(
        job["id"],
        finished_at=(datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
        cleanup_next_attempt_at=None,
    )
    service.cleanup_finished()
    assert service.store.job(job["id"])["container_removed_at"]
    service.gateway.remove_job.assert_called_once_with("container-1", job["id"])


def test_failed_retention_does_not_delay_successful_jobs(cleanup_service):
    service = cleanup_service
    service.settings.failed_container_retention_hours = 24
    job = finished_job(service)
    service.cleanup_finished()
    assert service.store.job(job["id"])["container_removed_at"]


def test_nonterminal_jobs_and_cancelled_queue_without_container_are_not_removed(cleanup_service):
    service = cleanup_service
    for job_id, state in (("queued-job", "queued"), ("running-job", "running"), ("starting-job", "starting")):
        service.store.create_job(job_id, "model_download", {}, None, job_id)
        service.store.update_job(job_id, status=state, container_id=f"container-{job_id}")
    cancelled = service.store.create_job("cancelled-queue", "model_download", {}, None, "cancelled")
    service.cancel(cancelled["id"])
    service.cleanup_finished()
    service.gateway.inspect_job.assert_not_called()
    service.gateway.stream_logs.assert_not_called()
    service.gateway.remove_job.assert_not_called()


def test_removed_container_is_not_processed_twice(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.cleanup_finished()
    assert service.store.job(job["id"])["container_removed_at"]
    service.gateway.reset_mock()
    service.cleanup_finished()
    service.gateway.inspect_job.assert_not_called()
    service.gateway.stream_logs.assert_not_called()
    service.gateway.remove_job.assert_not_called()


def test_tick_runs_terminal_cleanup_without_new_jobs(cleanup_service):
    service = cleanup_service
    job = finished_job(service, kind="model_download")
    service.tick()
    assert service.store.job(job["id"])["container_removed_at"]
    service.gateway.remove_job.assert_called_once_with("container-1", job["id"])
    service.gateway.run_worker.assert_not_called()


def test_archived_logs_and_metrics_remain_readable_without_remote_access(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.cleanup_finished()
    service.gateway.reset_mock()
    service.gateway.logs.side_effect = GatewayUnavailable("远端离线")
    service.gateway.read_file.side_effect = GatewayUnavailable("远端离线")
    service.gateway.run_worker.side_effect = GatewayUnavailable("远端离线")
    assert service.logs(job["id"], tail=1)["logs"].splitlines() == ["second line"]
    assert service.job(job["id"])["metrics"] == {"current_steps": 3, "loss": 0.25}
    service.gateway.logs.assert_not_called()
    service.gateway.read_file.assert_not_called()
    service.gateway.run_worker.assert_not_called()


def test_missing_local_archive_prevents_removal_when_rearchive_fails(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.store.update_job(job["id"], archived_at=timestamp(), metrics={"loss": 0.25})
    assert not console_path(service, job["id"]).exists()
    service.gateway.stream_logs.side_effect = GatewayUnavailable("重新归档日志失败")
    service.cleanup_finished()
    persisted = service.store.job(job["id"])
    assert persisted["status"] == "succeeded"
    assert not persisted["container_removed_at"]
    assert "重新归档日志失败" in persisted["cleanup_error"]
    service.gateway.stream_logs.assert_called_once_with("container-1")
    service.gateway.remove_job.assert_not_called()


def test_metrics_query_racing_with_cleanup_reads_newly_archived_metrics(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    archived_metrics = {"current_steps": 12, "loss": 0.125}

    def deleted_during_query(*_):
        service.store.update_job(
            job["id"], archived_at=timestamp(), container_removed_at=timestamp(), metrics=archived_metrics
        )
        raise ContainerMissing("查询时容器已经归档并删除")

    service.gateway.read_file.side_effect = deleted_during_query
    assert service.job(job["id"])["metrics"] == archived_metrics


def test_artifact_download_uses_reader_independent_of_removed_training_container(cleanup_service):
    service = cleanup_service
    job = finished_job(service)
    service.store.update_job(job["id"], archived_at=timestamp(), container_removed_at=timestamp())
    service.gateway.run_worker.return_value = {"artifacts": [{"path": "checkpoint-2", "is_dir": True}]}
    service.gateway.artifact_archive.return_value = iter([b"tar-header", b"tar-content"])
    assert b"".join(service.artifact_archive(job["id"], "checkpoint-2")) == b"tar-headertar-content"
    service.gateway.artifact_archive.assert_called_once_with(job["id"], "checkpoint-2")
    service.gateway.archive.assert_not_called()
    service.gateway.inspect_job.assert_not_called()


def test_database_migration_preserves_legacy_jobs_and_resources(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    old_time = "2026-09-24T10:00:00+00:00"
    payload = {"repo_id": "Qwen/test", "resource_id": "model-1"}
    result = {"path": "/lfa/models/model-1", "complete": True}
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT, error TEXT, container_id TEXT,
                idempotency_key TEXT UNIQUE, request_hash TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE resources (
                id TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (id, version)
            );
        """)
        connection.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "legacy-job",
                "model_download",
                "succeeded",
                json.dumps(payload),
                json.dumps(result),
                None,
                "legacy-container",
                "legacy-idempotency",
                "legacy-digest",
                0,
                old_time,
                old_time,
            ),
        )
        connection.execute(
            "INSERT INTO resources VALUES (?,?,?,?,?)",
            ("model-1", "1", "model", json.dumps(result), old_time),
        )
    store = Store(database)
    store.initialize()
    store.initialize()
    legacy = store.job("legacy-job")
    assert legacy["payload"] == payload
    assert legacy["result"] == result
    assert legacy["status"] == "succeeded"
    assert legacy["container_id"] == "legacy-container"
    assert legacy["created_at"] == old_time
    assert legacy["updated_at"] == old_time
    assert legacy["finished_at"] == old_time
    assert legacy["archived_at"] is None
    assert legacy["container_removed_at"] is None
    assert legacy["cleanup_error"] is None
    assert legacy["cleanup_next_attempt_at"] is None
    assert legacy["exit_code"] is None
    assert legacy["metrics"] is None
    assert store.by_key("legacy-idempotency")["id"] == "legacy-job"
    assert store.resource("model-1", "1") == result
