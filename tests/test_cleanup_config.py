"""验证容器归档回收的默认策略和环境配置约束。"""

import pytest
from pydantic import ValidationError

from llamafactory_agent.config import Settings


@pytest.fixture(autouse=True)
def clear_cleanup_environment(monkeypatch):
    monkeypatch.delenv("LFA_CLEANUP_INTERVAL", raising=False)
    monkeypatch.delenv("LFA_FAILED_CONTAINER_RETENTION_HOURS", raising=False)


def test_default_cleanup_has_no_failed_container_retention():
    settings = Settings(_env_file=None)

    assert settings.cleanup_interval == 30.0
    assert settings.failed_container_retention_hours == 0.0


def test_cleanup_policy_can_be_configured_by_environment(monkeypatch):
    monkeypatch.setenv("LFA_CLEANUP_INTERVAL", "5.5")
    monkeypatch.setenv("LFA_FAILED_CONTAINER_RETENTION_HOURS", "24")

    settings = Settings(_env_file=None)

    assert settings.cleanup_interval == 5.5
    assert settings.failed_container_retention_hours == 24.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"cleanup_interval": 0},
        {"cleanup_interval": 0.09},
        {"failed_container_retention_hours": -0.1},
    ],
)
def test_cleanup_policy_rejects_invalid_durations(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_cleanup_policy_accepts_minimum_durations():
    settings = Settings(_env_file=None, cleanup_interval=0.1, failed_container_retention_hours=0)

    assert settings.cleanup_interval == 0.1
    assert settings.failed_container_retention_hours == 0.0
