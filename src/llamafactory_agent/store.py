"""持久保存任务状态，避免控制端重启后丢失远端容器的对应关系。"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def timestamp() -> str:
    """生成可跨机器比较的任务时间。

    Returns:
        含 UTC 时区的 ISO 8601 时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


class Store:
    """每次操作使用独立 SQLite 连接，兼容 API 与调度线程并行访问。

    Args:
        path: 本地 SQLite 数据库路径。
    """

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """提供正常退出提交、异常退出回滚的事务连接。

        Yields:
            生命周期由上下文管理器维护的 SQLite 连接。
        """
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """建立本地数据库及幂等约束，不访问远端。

        Returns:
            无返回值。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, error TEXT, container_id TEXT,
                    idempotency_key TEXT UNIQUE, request_hash TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                );
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            for name, column_type in {
                "finished_at": "TEXT",
                "archived_at": "TEXT",
                "container_removed_at": "TEXT",
                "cleanup_error": "TEXT",
                "cleanup_next_attempt_at": "TEXT",
                "exit_code": "INTEGER",
                "metrics": "TEXT",
            }.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {column_type}")
            connection.execute(
                "UPDATE jobs SET finished_at=updated_at "
                "WHERE finished_at IS NULL AND status IN ('succeeded','failed','cancelled')"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS pending_cleanup ON jobs(cleanup_next_attempt_at) "
                "WHERE container_id IS NOT NULL AND container_removed_at IS NULL "
                "AND status IN ('succeeded','failed','cancelled')"
            )

    @staticmethod
    def decode_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """将数据库行转换为任务对象。

        Args:
            row: 查询所得任务行；不存在时为 None。
        Returns:
            已解码 JSON 和布尔字段的任务；空行返回 None。
        """
        if row is None:
            return None
        job = dict(row)
        for key in ("payload", "result", "metrics"):
            job[key] = json.loads(job[key]) if job[key] else None
        job["cancel_requested"] = bool(job["cancel_requested"])
        return job

    def create_job(self, job_id: str, kind: str, payload: dict, key: str | None, digest: str) -> dict:
        """创建排队任务，以数据库唯一约束保护幂等键。

        Args:
            job_id: 任务唯一标识。
            kind: 下载、导入或训练任务类型。
            payload: 已规范化的任务参数。
            key: 可选的请求幂等键。
            digest: 任务参数摘要。
        Returns:
            已持久化的排队任务。
        """
        now = timestamp()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO jobs (id,kind,status,payload,idempotency_key,request_hash,created_at,updated_at) "
                "VALUES (?,?, 'queued', ?,?,?,?,?)",
                (job_id, kind, json.dumps(payload, ensure_ascii=False), key, digest, now, now),
            )
        return self.job(job_id)

    def job(self, job_id: str) -> dict:
        """读取任务记录；不存在时抛出 KeyError。

        Args:
            job_id: 任务唯一标识。
        Returns:
            完整任务记录。
        """
        with self.connection() as connection:
            job = self.decode_job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        if job is None:
            raise KeyError(job_id)
        return job

    def by_key(self, key: str) -> dict | None:
        """读取幂等键对应的任务。

        Args:
            key: 请求幂等键。
        Returns:
            已有任务；不存在时返回 None。
        """
        with self.connection() as connection:
            return self.decode_job(
                connection.execute("SELECT * FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
            )

    def jobs(self, active_only: bool = False) -> list[dict]:
        """按提交顺序列出任务。

        Args:
            active_only: 是否仅返回尚未终结的任务。
        Returns:
            满足筛选条件的任务列表。
        """
        query = "SELECT * FROM jobs"
        if active_only:
            query += " WHERE status NOT IN ('succeeded','failed','cancelled')"
        query += " ORDER BY created_at"
        with self.connection() as connection:
            return [self.decode_job(row) for row in connection.execute(query)]

    def update_job(self, job_id: str, **changes: Any) -> dict:
        """更新任务状态，同时维护最后修改时间。

        Args:
            job_id: 任务唯一标识。
            **changes: 允许修改的状态、参数、结果或容器信息。
        Returns:
            更新后的完整任务记录。
        """
        allowed = {
            "status",
            "payload",
            "result",
            "error",
            "container_id",
            "cancel_requested",
            "finished_at",
            "archived_at",
            "container_removed_at",
            "cleanup_error",
            "cleanup_next_attempt_at",
            "exit_code",
            "metrics",
        }
        if not changes.keys() <= allowed:
            raise ValueError("不允许修改任务标识或幂等键")
        changes["updated_at"] = timestamp()
        for key in ("payload", "result", "metrics"):
            if key in changes:
                changes[key] = json.dumps(changes[key], ensure_ascii=False)
        with self.connection() as connection:
            assignments = ",".join(f"{key}=?" for key in changes)
            if changes.get("status") in TERMINAL_STATES and "finished_at" not in changes:
                assignments += ",finished_at=COALESCE(finished_at,?)"
                values = (*changes.values(), changes["updated_at"], job_id)
            else:
                values = (*changes.values(), job_id)
            connection.execute(f"UPDATE jobs SET {assignments} WHERE id=?", values)
        return self.job(job_id)

    def pending_cleanup(self) -> list[dict]:
        """查询已到回收时间的终态任务，保留失败后的重试计划。

        Returns:
            尚未确认容器移除、且已到重试时间的任务记录。
        """
        with self.connection() as connection:
            return [
                self.decode_job(row)
                for row in connection.execute(
                    "SELECT * FROM jobs WHERE status IN ('succeeded','failed','cancelled') "
                    "AND container_id IS NOT NULL AND container_removed_at IS NULL "
                    "AND (cleanup_next_attempt_at IS NULL OR cleanup_next_attempt_at<=?) "
                    "ORDER BY finished_at",
                    (timestamp(),),
                )
            ]

    def save_resource(self, kind: str, resource_id: str, version: str, payload: dict) -> dict:
        """保存资源版本及其可供训练引用的元信息。

        Args:
            kind: 模型或数据集资源类型。
            resource_id: 资源唯一标识。
            version: 资源版本标识。
            payload: 远端资源的元信息。
        Returns:
            包含标识、版本和类型的完整资源对象。
        """
        resource = {**payload, "id": resource_id, "version": version, "kind": kind}
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO resources (id,version,kind,payload,created_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id,version) DO UPDATE SET payload=excluded.payload",
                (resource_id, version, kind, json.dumps(resource, ensure_ascii=False), timestamp()),
            )
        return resource

    def resource(self, resource_id: str, version: str | None = None) -> dict:
        """读取资源版本；不存在时抛出 KeyError。

        Args:
            resource_id: 资源唯一标识。
            version: 可选版本；省略时读取最新登记版本。
        Returns:
            完整资源对象。
        """
        with self.connection() as connection:
            if version:
                row = connection.execute(
                    "SELECT payload FROM resources WHERE id=? AND version=?", (resource_id, version)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT payload FROM resources WHERE id=? ORDER BY created_at DESC LIMIT 1",
                    (resource_id,),
                ).fetchone()
        if row is None:
            raise KeyError(resource_id)
        return json.loads(row["payload"])

    def resources(self, kind: str) -> list[dict]:
        """按登记时间倒序列出资源版本。

        Args:
            kind: 模型或数据集资源类型。
        Returns:
            指定类型的全部资源版本。
        """
        with self.connection() as connection:
            return [
                json.loads(row["payload"])
                for row in connection.execute(
                    "SELECT payload FROM resources WHERE kind=? ORDER BY created_at DESC", (kind,)
                )
            ]
