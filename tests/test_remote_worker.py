"""使用小型本地资源验证远端工具协议，不连接 GPU 或下载模型。"""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llamafactory_agent import remote_worker as worker


@pytest.fixture
def managed_root(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    return tmp_path


def make_model(path: Path, adapter: bool = False) -> Path:
    """构建无需加载权重的模型目录。

    Args:
        path: 测试目录。
        adapter: 是否构建适配器。
    Returns:
        创建的模型目录。
    """
    path.mkdir(parents=True)
    config = "adapter_config.json" if adapter else "config.json"
    (path / config).write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (path / ("adapter_model.safetensors" if adapter else "model.safetensors")).write_bytes(b"weight")
    if not adapter:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_model_requires_all_index_shards(tmp_path):
    model = make_model(tmp_path / "model")
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"first": "first.safetensors", "second": "second.safetensors"}})
    )
    (model / "first.safetensors").write_bytes(b"first")
    inspected = worker.inspect_model(model)
    assert not inspected["complete"]
    assert inspected["missing"] == ["second.safetensors"]
    (model / "second.safetensors").write_bytes(b"second")
    assert worker.inspect_model(model)["complete"]


def test_adapter_is_not_reported_as_base(tmp_path):
    inspected = worker.inspect_model(make_model(tmp_path / "adapter", adapter=True))
    assert inspected["kind"] == "adapter"
    assert inspected["complete"]


def test_invalid_index_and_tokenizer_are_incomplete(tmp_path):
    model = make_model(tmp_path / "model")
    (model / "tokenizer.json").unlink()
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../escape"}}))
    inspected = worker.inspect_model(model)
    assert not inspected["complete"]
    assert "tokenizer" in inspected["missing"]
    assert "model.safetensors.index.json: 无效权重索引" in inspected["missing"]


def test_catalog_does_not_substitute_huggingface(monkeypatch):
    constants = SimpleNamespace(
        DownloadSource=SimpleNamespace(MODELSCOPE="ms"),
        SUPPORTED_MODELS={"only-hf": {"hf": "upstream/model"}, "both": {"hf": "hf/model", "ms": "ms/model"}},
        DEFAULT_TEMPLATE={"both": "qwen"},
    )
    monkeypatch.setattr(worker.importlib, "import_module", lambda _: constants)
    entries = worker.catalog({})["models"]
    assert entries[0]["repo_id"] is None
    assert entries[0]["status"] == "unsupported_source"
    assert entries[1]["repo_id"] == "ms/model"


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_import_normalizes_and_versions_local_data(managed_root, suffix):
    source = managed_root / f"source.{suffix}"
    records = [{"instruction": "问题", "output": "回答"}, {"instruction": "下一题", "output": "下一答"}]
    source.write_text(
        json.dumps(records, ensure_ascii=False)
        if suffix == "json"
        else "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    payload = {"resource_id": "dataset", "version": "v1", "source_path": str(source)}
    manifest = worker.dataset_import(payload)
    assert manifest["record_count"] == 2
    assert manifest["preview"][0]["input"] == ""
    assert len(manifest["sha256"]) == 64
    assert Path(manifest["path"]).is_file()
    assert worker.dataset_preview({"path": manifest["path"], "limit": 1})["count"] == 1
    with pytest.raises(ValueError, match="版本已存在"):
        worker.dataset_import(payload)


def test_streaming_json_array_accepts_large_record_and_rejects_trailing_comma():
    text = json.dumps([{"output": "中文" * 70000}, {"output": "结束"}], ensure_ascii=False)
    assert len(list(worker._json_array(io.StringIO(text)))) == 2
    with pytest.raises(ValueError):
        list(worker._json_array(io.StringIO('[{"value":1},]')))
    with pytest.raises(ValueError):
        list(worker._json_array(io.StringIO('[{"value":1}] trailing')))


def test_bad_record_does_not_publish_manifest(managed_root):
    source = managed_root / "source.jsonl"
    source.write_text('{"instruction":"缺少回答"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 条记录"):
        worker.dataset_import({"resource_id": "dataset", "version": "v1", "source_path": str(source)})
    assert not (managed_root / "datasets/dataset/v1/manifest.json").exists()
    assert not (managed_root / "datasets/dataset/v1/data.jsonl").exists()


@pytest.mark.parametrize("contents", ["", "[]", "\n \n"])
def test_empty_dataset_is_rejected(managed_root, contents):
    source = managed_root / "empty.json"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="数据集不能为空"):
        worker.dataset_import({"resource_id": "empty", "version": "v1", "source_path": str(source)})
    assert not (managed_root / "datasets/empty/v1/manifest.json").exists()


def test_scan_supports_explicit_external_roots_and_depth(managed_root):
    external = managed_root / "external"
    make_model(external / "available")
    make_model(external / "deep/other/hidden")
    result = worker.scan_models({"roots": [str(external)], "max_depth": 1})
    assert len(result["models"]) == 1
    assert result["models"][0]["path"] == str(external / "available")


def test_sharegpt_preference_mapping(managed_root):
    source = managed_root / "preference.json"
    source.write_text(
        json.dumps(
            [
                {
                    "messages": [{"role": "user", "content": "问题"}],
                    "chosen": {"role": "assistant", "content": "好的回答"},
                    "rejected": {"role": "assistant", "content": "较差回答"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = worker.dataset_import(
        {
            "resource_id": "preferences",
            "version": "v1",
            "source_path": str(source),
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    )
    assert manifest["record_count"] == 1
    assert manifest["tags"]["system_tag"] == "system"


def test_dataset_catalog_ignores_all_online_entries(managed_root):
    source = managed_root / "local.json"
    source.write_text("[]")
    index = managed_root / "dataset_info.json"
    index.write_text(
        json.dumps(
            {
                "local": {"file_name": "local.json"},
                "hf": {"hf_hub_url": "owner/data"},
                "ms": {"ms_hub_url": "owner/data"},
            }
        )
    )
    result = worker.datasets_catalog({"path": str(index)})
    assert result["ignored_online_count"] == 2
    assert result["datasets"][0]["exists"]
    assert len(result["datasets"]) == 1


def test_download_uses_modelscope_only_and_records_revision(managed_root, monkeypatch):
    api = Mock()
    api.get_valid_revision_detail.return_value = {"Revision": "v1.0", "CreatedAt": 1}
    api.get_model_files.return_value = [
        {"Type": "blob", "Path": "model.safetensors", "Sha256": "abc", "Size": 6}
    ]

    def snapshot_download(repository, **kwargs):
        assert repository == "Qwen/Qwen2.5-0.5B-Instruct"
        assert kwargs["revision"] == "v1.0"
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("{}")
        (target / "tokenizer.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"weight")
        return str(target)

    modules = {
        "modelscope": SimpleNamespace(snapshot_download=snapshot_download),
        "modelscope.hub.api": SimpleNamespace(HubApi=lambda: api),
    }
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: modules[name])
    manifest = worker.model_download({"resource_id": "base", "repo_id": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert manifest["source"] == "modelscope"
    assert manifest["revision"] == "v1.0"
    assert (managed_root / "models/base/manifest.json").exists()
    assert worker.scan_models({})["models"][0]["resource_id"] == "base"


def test_prepare_train_protects_paths_and_disables_v1(managed_root):
    model = make_model(managed_root / "models/base")
    source = managed_root / "data.jsonl"
    source.write_text('{"instruction":"问题","output":"回答"}\n', encoding="utf-8")
    payload = {
        "job_id": "job",
        "model_path": str(model),
        "dataset": {"path": str(source), "formatting": "alpaca"},
        "training_args": {"stage": "sft", "do_train": True, "template": "qwen"},
    }
    result = worker.prepare_train(payload)
    configuration = json.loads(Path(result["config_path"]).read_text())
    assert result["environment"]["USE_V1"] == "0"
    assert result["environment"]["HF_HUB_OFFLINE"] == "1"
    assert configuration["dataset"] == "training_data"
    assert configuration["model_name_or_path"] == str(model)
    assert configuration["output_dir"] == str(managed_root / "jobs/job/output")
    assert configuration["cache_dir"] == str(managed_root / "jobs/job/cache")
    assert Path(configuration["cache_dir"]).is_dir()
    assert worker.prepare_train(payload) == result
    assert json.loads(Path(result["config_path"]).read_text()) == configuration
    payload["training_args"]["dataset_dir"] = "ONLINE"
    with pytest.raises(ValueError, match="服务端管理"):
        worker.prepare_train(payload)


def test_checkpoint_needs_optimizer_scheduler_and_weights(managed_root):
    checkpoint = make_model(managed_root / "jobs/first/output/checkpoint-2", adapter=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    assert not worker.check_checkpoint({"path": str(checkpoint)})["valid"]
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    assert worker.check_checkpoint({"path": str(checkpoint)})["valid"]
    artifacts = worker.list_artifacts({"job_id": "first"})["artifacts"]
    assert any(entry["path"] == "checkpoint-2/optimizer.pt" for entry in artifacts)


def test_preflight_detects_deleted_model_without_writes(managed_root):
    model = make_model(managed_root / "models/base")
    source = managed_root / "data.jsonl"
    source.write_text('{"instruction":"问题","output":"回答"}\n', encoding="utf-8")
    payload = {"model_path": str(model), "dataset": {"path": str(source)}, "training_args": {"max_steps": -1}}
    initial_paths = sorted(str(path) for path in managed_root.rglob("*"))
    assert worker.preflight(payload)["valid"]
    assert sorted(str(path) for path in managed_root.rglob("*")) == initial_paths
    (model / "model.safetensors").unlink()
    result = worker.preflight(payload)
    assert not result["valid"]
    assert "model_weights" in result["errors"][0]
    (model / "model.safetensors").write_bytes(b"restored")
    assert worker.preflight(payload)["valid"]


def test_metrics_reads_large_log_tail_and_ignores_partial_final_line(managed_root):
    output = managed_root / "jobs/training/output"
    output.mkdir(parents=True)
    log = output / "trainer_log.jsonl"
    historical_line = (
        json.dumps({"loss": 2.0, "description": "历史" * 500}, ensure_ascii=False).encode() + b"\n"
    )
    with log.open("wb") as destination:
        for _ in range(1000):
            destination.write(historical_line)
        destination.write(b'{"current_steps":1000,"loss":0.25}\n')
        destination.write(b'{"current_steps":1001,"loss":')
    assert log.stat().st_size > 2 * 1024 * 1024
    assert worker.read_metrics({"job_id": "training"}) == {"metrics": {"current_steps": 1000, "loss": 0.25}}
    with log.open("ab") as destination:
        destination.write(b"0.2}\n")
    assert worker.read_metrics({"job_id": "training"})["metrics"]["current_steps"] == 1001


def test_metrics_missing_or_only_partial_log_returns_none(managed_root):
    assert worker.read_metrics({"job_id": "missing"}) == {"metrics": None}
    output = managed_root / "jobs/partial/output"
    output.mkdir(parents=True)
    (output / "trainer_log.jsonl").write_bytes(b'{"loss":0.5}')
    assert worker.read_metrics({"job_id": "partial"}) == {"metrics": None}


@pytest.mark.parametrize(
    "name,value",
    [
        ("cutoff_len", 0),
        ("per_device_train_batch_size", -1),
        ("gradient_accumulation_steps", True),
        ("max_steps", -2),
    ],
)
def test_preflight_rejects_invalid_numeric_parameters(managed_root, name, value):
    model = make_model(managed_root / "models/base")
    source = managed_root / "data.jsonl"
    source.write_text("{}")
    result = worker.preflight(
        {"model_path": str(model), "dataset": {"path": str(source)}, "training_args": {name: value}}
    )
    assert not result["valid"]
    assert any(name in error for error in result["errors"])


def test_resource_id_cannot_escape_managed_root(managed_root):
    with pytest.raises(ValueError, match="资源标识"):
        worker.model_download({"resource_id": "../escape", "repo_id": "Qwen/model"})


def test_protocol_emits_structured_error(capsys):
    assert worker.main(["unsupported", "{}"]) == 1
    line = capsys.readouterr().out.strip()
    assert line.startswith(worker.RESULT_PREFIX)
    assert json.loads(line[len(worker.RESULT_PREFIX) :])["error"]["type"] == "ValueError"
