"""在现有训练镜像内运行的一次性工具，不依赖常驻服务。"""

from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO

ROOT = Path("/lfa")
RESULT_PREFIX = "LFA_RESULT="
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
MODEL_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
DEFAULT_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected",
}
DEFAULT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
    "system_tag": "system",
}
CONTROLLED_TRAINING_ARGUMENTS = {
    "model_name_or_path",
    "dataset",
    "dataset_dir",
    "output_dir",
    "resume_from_checkpoint",
    "eval_dataset",
    "tokenized_path",
    "adapter_name_or_path",
    "cache_dir",
    "logging_dir",
}


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("资源标识只能包含字母、数字、下划线、连字符和点，且必须以字母或数字开头")
    return value


def _write_json(path: Path, value: Any) -> None:
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _local_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or "://" in value:
        raise ValueError("必须使用容器内的绝对文件路径，不能引用在线仓库")
    return path


def initialize(payload: dict[str, Any]) -> dict[str, Any]:
    """初始化受管存储。

    Args:
        payload: 保留的操作参数。
    Returns:
        容器中的存储根路径。
    """
    for directory in ("models", "datasets", "cache", "jobs", "incoming"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    return {"root": str(ROOT)}


def resources(payload: dict[str, Any]) -> dict[str, Any]:
    """查询容器可见 GPU 与磁盘容量。

    Args:
        payload: 保留的操作参数。
    Returns:
        GPU 列表、磁盘容量和可选的 GPU 查询错误。
    """
    fields = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    gpu_list: list[dict[str, Any]] = []
    gpu_error = None
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            gpu_list.append(
                {
                    "index": int(row[0]),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": int(row[3]),
                    "memory_used_mib": int(row[4]),
                    "memory_free_mib": int(row[5]),
                    "utilization_percent": int(row[6]) if row[6].strip().isdigit() else None,
                }
            )
    except (OSError, subprocess.SubprocessError) as error:
        gpu_error = str(error)
    disk = None
    if ROOT.exists():
        stats = os.statvfs(ROOT)
        disk = {
            "path": str(ROOT),
            "total_bytes": stats.f_blocks * stats.f_frsize,
            "free_bytes": stats.f_bavail * stats.f_frsize,
        }
    return {"gpus": gpu_list, "disk": disk, "gpu_error": gpu_error}


def catalog(payload: dict[str, Any]) -> dict[str, Any]:
    """从当前镜像的模型预设提取魔搭下载候选。

    Args:
        payload: 保留的操作参数。
    Returns:
        包含魔搭映射状态的模型清单。
    """
    constants = importlib.import_module("llamafactory.extras.constants")
    source = constants.DownloadSource.MODELSCOPE
    models = []
    for name, sources in constants.SUPPORTED_MODELS.items():
        repository = sources.get(source)
        models.append(
            {
                "name": name,
                "repo_id": repository,
                "source": "modelscope",
                "template": constants.DEFAULT_TEMPLATE.get(name, ""),
                "status": "available" if repository else "unsupported_source",
            }
        )
    return {"models": models}


def inspect_model(directory: str | Path) -> dict[str, Any]:
    """只读检查模型配置、分词器和权重分片的完整性。

    Args:
        directory: 本地模型或 adapter 目录。
    Returns:
        模型类型、完整性、缺失文件与权重大小；不加载权重或执行模型代码。
    """
    root = Path(directory)
    adapter = (root / "adapter_config.json").is_file()
    config_name = "adapter_config.json" if adapter else "config.json"
    missing: list[str] = []
    configuration: dict[str, Any] = {}
    try:
        configuration = _read_json(root / config_name)
        if not isinstance(configuration, dict):
            missing.append(f"{config_name}: 非对象配置")
            configuration = {}
    except (OSError, ValueError):
        missing.append(config_name)

    if not adapter:
        tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt")
        tokenizer_exists = any(
            (root / filename).is_file() and (root / filename).stat().st_size for filename in tokenizer_files
        )
        byte_pair_exists = all(
            (root / filename).is_file() and (root / filename).stat().st_size
            for filename in ("vocab.json", "merges.txt")
        )
        if not tokenizer_exists and not byte_pair_exists:
            missing.append("tokenizer")

    indexes = sorted(root.glob("*.index.json"))
    weight_files: set[Path] = set()
    for index in indexes:
        try:
            mapping = _read_json(index).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("缺少 weight_map")
            for filename in set(mapping.values()):
                shard = root / filename
                if (
                    not isinstance(filename, str)
                    or Path(filename).is_absolute()
                    or ".." in Path(filename).parts
                ):
                    raise ValueError("分片路径越界")
                if not shard.is_file() or shard.stat().st_size == 0:
                    missing.append(filename)
                else:
                    weight_files.add(shard)
        except (OSError, TypeError, AttributeError, ValueError):
            missing.append(f"{index.name}: 无效权重索引")

    single_weights = (
        ("adapter_model.safetensors", "adapter_model.bin")
        if adapter
        else ("model.safetensors", "pytorch_model.bin")
    )
    for filename in single_weights:
        weight = root / filename
        if weight.is_file() and weight.stat().st_size:
            weight_files.add(weight)
    if not weight_files and not indexes:
        missing.append("adapter_weights" if adapter else "model_weights")

    inspection = {
        "path": str(root),
        "kind": "adapter" if adapter else "model",
        "status": "incomplete" if missing else "complete",
        "complete": not missing,
        "missing": sorted(set(missing)),
        "model_type": configuration.get("model_type"),
        "architectures": configuration.get("architectures", []),
        "parameter_size_bytes": sum(weight.stat().st_size for weight in weight_files),
    }
    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            stored = _read_json(manifest)
            for key in (
                "resource_id",
                "repo_id",
                "source",
                "revision",
                "requested_revision",
                "snapshot_sha256",
            ):
                if key in stored:
                    inspection[key] = stored[key]
        except (OSError, ValueError, TypeError):
            pass
    return inspection


def scan_models(payload: dict[str, Any]) -> dict[str, Any]:
    """按有限深度扫描受管与外部模型目录。

    Args:
        payload: roots，或 root 与 external_paths，以及可选的 max_depth。
    Returns:
        已发现模型、缺失目录及是否达到目录数量上限。
    """
    roots = payload.get("roots") or [
        payload.get("root", str(ROOT / "models")),
        *payload.get("external_paths", []),
    ]
    max_depth = min(max(int(payload.get("max_depth", 3)), 0), 5)
    models = []
    missing_roots = []
    visited = set()
    directory_count = 0
    for raw_root in roots:
        root = _local_path(raw_root)
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        for directory, children, filenames in os.walk(root, followlinks=False):
            directory_count += 1
            if directory_count > 5000:
                return {"models": models, "missing_roots": missing_roots, "truncated": True}
            current = Path(directory)
            depth = len(current.relative_to(root).parts)
            children[:] = (
                sorted(
                    child
                    for child in children
                    if not child.startswith(".") and not (current / child).is_symlink()
                )
                if depth < max_depth
                else []
            )
            if "config.json" not in filenames and "adapter_config.json" not in filenames:
                continue
            resolved = str(current.resolve())
            if resolved not in visited:
                models.append(inspect_model(current))
                visited.add(resolved)
            children[:] = []
    return {"models": models, "missing_roots": missing_roots, "truncated": False}


def model_download(payload: dict[str, Any]) -> dict[str, Any]:
    """仅使用魔搭下载并登记独立模型快照。

    Args:
        payload: resource_id、魔搭 repo_id 及可选 revision。
    Returns:
        完整模型清单，包含实际版本名与上游文件清单摘要。
    """
    identifier = _identifier(payload["resource_id"])
    repository = payload["repo_id"]
    if not MODEL_REPOSITORY.fullmatch(repository):
        raise ValueError("魔搭模型标识必须采用组织名/模型名格式")
    target = ROOT / "models" / identifier
    if (target / "manifest.json").exists():
        raise ValueError("模型资源已存在，下载新版本必须创建新的资源标识")
    sdk = importlib.import_module("modelscope")
    api_module = importlib.import_module("modelscope.hub.api")
    api = api_module.HubApi()
    requested_revision = payload.get("revision")
    revision_detail = api.get_valid_revision_detail(repository, revision=requested_revision)
    revision = revision_detail["Revision"]
    files = api.get_model_files(repository, revision=revision, recursive=True)
    snapshot_files = [
        {key: entry.get(key) for key in ("Path", "Revision", "Sha256", "Size")}
        for entry in files
        if entry.get("Type") != "tree"
    ]
    snapshot_files.sort(key=lambda entry: entry["Path"] or "")
    snapshot_sha256 = hashlib.sha256(json.dumps(snapshot_files, sort_keys=True).encode()).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    sdk.snapshot_download(
        repository, revision=revision, local_dir=str(target), cache_dir=str(ROOT / "cache" / "modelscope")
    )
    manifest = inspect_model(target)
    if not manifest["complete"] or manifest["kind"] != "model":
        raise ValueError(
            f"模型下载后未通过完整基座检查：{', '.join(manifest['missing']) or manifest['kind']}"
        )
    manifest.update(
        {
            "resource_id": identifier,
            "repo_id": repository,
            "source": "modelscope",
            "requested_revision": requested_revision,
            "revision": revision,
            "revision_detail": revision_detail,
            "snapshot_sha256": snapshot_sha256,
        }
    )
    _write_json(target / "manifest.json", manifest)
    return manifest


def _json_array(stream: TextIO) -> Iterator[Any]:
    # 标准库不提供流式数组解析；按单条记录解码，避免整份训练数据驻留内存。
    decoder = json.JSONDecoder()
    buffer = ""
    finished = False

    def refill() -> bool:
        nonlocal buffer, finished
        chunk = stream.read(65536)
        buffer += chunk
        finished = not chunk
        return bool(chunk)

    def trim() -> None:
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not finished:
            refill()
            buffer = buffer.lstrip()

    trim()
    if not buffer.startswith("["):
        raise ValueError("JSON 数据集必须是记录数组")
    buffer = buffer[1:]
    trim()
    if buffer.startswith("]"):
        buffer = buffer[1:]
    else:
        while True:
            trim()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if not refill():
                        raise ValueError("JSON 数据集存在无效或未闭合记录") from error
            yield value
            buffer = buffer[end:]
            trim()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not buffer.startswith(","):
                raise ValueError("JSON 数组记录之间缺少逗号")
            buffer = buffer[1:]
    trim()
    if buffer:
        raise ValueError("JSON 数组后存在额外内容")


def iter_records(source: Path) -> Iterator[dict[str, Any]]:
    """逐条读取 UTF-8 JSON 数组或 JSONL 文件。

    Args:
        source: 本地数据文件。
    Yields:
        单条字典记录；输入非法时抛出包含记录位置的错误。
    """
    with source.open(encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            character = stream.read(1)
            if not character:
                return
            first = character.strip()
        stream.seek(0)
        if first == "[":
            records = _json_array(stream)
        else:
            records = (json.loads(line) for line in stream if line.strip())
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError(f"第 {index} 条记录必须是 JSON 对象")
            yield record


def _required_string(record: dict[str, Any], field: str | None) -> None:
    if field is None or not isinstance(record.get(field), str):
        raise ValueError(f"字段 {field!r} 必须存在且为字符串")


def _validate_record(record: dict[str, Any], manifest: dict[str, Any]) -> None:
    columns, tags = manifest["columns"], manifest["tags"]
    ranking = manifest["ranking"]
    for role in ("system", "tools"):
        if columns.get(role):
            _required_string(record, columns[role])
    if manifest["formatting"] == "alpaca":
        prompt_field = columns.get("prompt")
        query_field = columns.get("query")
        if not prompt_field and not query_field:
            raise ValueError("指令数据必须至少映射 prompt 或 query 字段")
        if prompt_field:
            _required_string(record, prompt_field)
        if query_field:
            if prompt_field:
                record.setdefault(query_field, "")
            _required_string(record, query_field)
        for field in ("chosen", "rejected") if ranking else ("response",):
            _required_string(record, columns.get(field))
        history_field = columns.get("history")
        if history_field and not (
            isinstance(record.get(history_field), list)
            and all(
                isinstance(pair, list) and len(pair) == 2 and all(isinstance(turn, str) for turn in pair)
                for pair in record[history_field]
            )
        ):
            raise ValueError("history 必须是用户与助手文本组成的二元数组列表")
        return
    messages = record.get(columns["messages"])
    if not isinstance(messages, list) or not messages:
        raise ValueError("对话字段必须是非空消息数组")
    role_field, content_field = tags["role_tag"], tags["content_tag"]
    if isinstance(messages[0], dict) and messages[0].get(role_field) == tags["system_tag"]:
        _required_string(messages[0], content_field)
        messages = messages[1:]
    if not messages or len(messages) % 2 != (1 if ranking else 0):
        raise ValueError("对话轮次不合法：普通对话须以助手结束，偏好对话须以用户结束")
    for index, message in enumerate(messages):
        expected_roles = (
            (tags["user_tag"], tags["observation_tag"])
            if index % 2 == 0
            else (tags["assistant_tag"], tags["function_tag"])
        )
        if not isinstance(message, dict) or message.get(role_field) not in expected_roles:
            raise ValueError(f"第 {index + 1} 条消息角色或顺序不合法")
        _required_string(message, content_field)
    if ranking:
        for role in ("chosen", "rejected"):
            message = record.get(columns[role])
            if not isinstance(message, dict) or message.get(role_field) not in (
                tags["assistant_tag"],
                tags["function_tag"],
            ):
                raise ValueError(f"{role} 必须是助手消息对象")
            _required_string(message, content_field)


def dataset_import(payload: dict[str, Any]) -> dict[str, Any]:
    """验证本地文本数据并保存不可覆盖的 JSONL 版本。

    Args:
        payload: 标识、版本、source_path、formatting、columns、tags 和 ranking。
    Returns:
        包含数据路径、记录数、摘要和少量预览的版本清单。
    """
    identifier, version = _identifier(payload["resource_id"]), _identifier(payload["version"])
    source = _local_path(payload["source_path"])
    formatting = payload.get("formatting", "alpaca")
    if formatting not in ("alpaca", "sharegpt"):
        raise ValueError("首版仅支持 alpaca 和 sharegpt 文本格式")
    supplied_columns = payload.get("columns") or {}
    if any(supplied_columns.get(key) for key in ("images", "videos", "audios")):
        raise ValueError("首版不支持多模态数据")
    target = ROOT / "datasets" / identifier / version
    if target.exists():
        raise ValueError("数据集版本已存在，请创建新版本")
    target.mkdir(parents=True)
    output = target / "data.jsonl"
    pending = target / "data.jsonl.partial"
    manifest = {
        "resource_id": identifier,
        "version": version,
        "path": str(output),
        "formatting": formatting,
        "columns": {**DEFAULT_COLUMNS, **supplied_columns},
        "tags": {**DEFAULT_TAGS, **(payload.get("tags") or {})},
        "ranking": bool(payload.get("ranking", False)),
        "record_count": 0,
        "preview": [],
    }
    # 完整映射避免上游 parser 在提供部分 columns/tags 后把默认字段置为空。
    if formatting == "alpaca":
        manifest["columns"].pop("messages", None)
    else:
        for key in ("prompt", "query", "response"):
            manifest["columns"].pop(key, None)
    if not manifest["ranking"]:
        manifest["columns"].pop("chosen", None)
        manifest["columns"].pop("rejected", None)
    digest = hashlib.sha256()
    with pending.open("wb") as destination:
        for index, record in enumerate(iter_records(source), 1):
            try:
                _validate_record(record, manifest)
            except ValueError as error:
                raise ValueError(f"第 {index} 条记录：{error}") from error
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            destination.write(encoded)
            digest.update(encoded)
            manifest["record_count"] += 1
            if len(manifest["preview"]) < 3:
                manifest["preview"].append(record)
    if not manifest["record_count"]:
        raise ValueError("数据集不能为空")
    pending.replace(output)
    manifest["sha256"] = digest.hexdigest()
    _write_json(target / "manifest.json", manifest)
    return manifest


def dataset_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """读取有限数量的本地数据样本。

    Args:
        payload: path 与 limit，limit 最大为 20。
    Returns:
        items 样本数组和实际 count。
    """
    limit = min(max(int(payload.get("limit", payload.get("count", 3))), 1), 20)
    items = list(itertools.islice(iter_records(_local_path(payload["path"])), limit))
    return {"items": items, "count": len(items)}


def datasets_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    """只登记现有索引中的本地文件，不访问任何数据集 Hub。

    Args:
        payload: 可选的 dataset_info.json 绝对路径。
    Returns:
        本地条目与被忽略的在线条目数量。
    """
    index_path = _local_path(payload.get("path", "/source/dataset_info.json"))
    if not index_path.is_file():
        return {"datasets": [], "ignored_online_count": 0, "exists": False}
    datasets = []
    ignored = 0
    for name, entry in _read_json(index_path).items():
        filename = entry.get("file_name")
        if not filename:
            ignored += 1
            continue
        path = index_path.parent / filename
        datasets.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "formatting": entry.get("formatting", "alpaca"),
                "columns": entry.get("columns", {}),
                "tags": entry.get("tags", {}),
                "ranking": entry.get("ranking", False),
            }
        )
    return {"datasets": datasets, "ignored_online_count": ignored, "exists": True}


def list_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """枚举训练输出，禁止符号链接指向输出目录外部。

    Args:
        payload: 包含 job_id 的参数。
    Returns:
        相对路径、大小及目录标志，最多返回 10000 个条目。
    """
    output = ROOT / "jobs" / _identifier(payload["job_id"]) / "output"
    if not output.is_dir():
        return {"artifacts": [], "truncated": False}
    artifacts = []
    resolved_output = output.resolve()
    for directory, children, filenames in os.walk(output, followlinks=False):
        current = Path(directory)
        depth = len(current.relative_to(output).parts)
        children[:] = (
            sorted(child for child in children if not (current / child).is_symlink()) if depth < 5 else []
        )
        for name in sorted([*children, *filenames]):
            path = current / name
            if path.is_symlink() or not path.resolve().is_relative_to(resolved_output):
                continue
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "size_bytes": 0 if path.is_dir() else path.stat().st_size,
                    "is_dir": path.is_dir(),
                }
            )
            if len(artifacts) >= 10000:
                return {"artifacts": artifacts, "truncated": True}
    return {"artifacts": artifacts, "truncated": False}


def read_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """从训练日志尾部读取最后一条已经完整写入的指标。

    Args:
        payload: job_id 和可选的 max_bytes，上限为 64 KiB。
    Returns:
        metrics 为最后一个合法 JSON 对象，无完整记录时为 None。
    """
    log_path = ROOT / "jobs" / _identifier(payload["job_id"]) / "output" / "trainer_log.jsonl"
    limit = min(max(int(payload.get("max_bytes", 65536)), 1), 65536)
    try:
        with log_path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            offset = max(0, size - limit)
            source.seek(offset)
            tail = source.read(limit)
    except FileNotFoundError:
        return {"metrics": None}
    if offset:
        # 窗口起点可能落在多字节字符或长记录中，首行不参与 JSON 解析。
        tail = tail.partition(b"\n")[2]
    if b"\n" not in tail:
        return {"metrics": None}
    complete = tail.rsplit(b"\n", 1)[0]
    for line in reversed(complete.splitlines()):
        try:
            metric = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(metric, dict):
            return {"metrics": metric}
    return {"metrics": None}


def check_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """检查普通单 GPU 训练断点的权重和优化器状态。

    Args:
        payload: 包含本地 checkpoint path 的参数。
    Returns:
        valid 和缺失文件列表；不将仅有 adapter 权重视作完整断点。
    """
    checkpoint = _local_path(payload["path"])
    if not checkpoint.resolve().is_relative_to((ROOT / "jobs").resolve()):
        raise ValueError("检查点必须位于受管任务目录内")
    missing = [
        filename
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if not (checkpoint / filename).is_file() or (checkpoint / filename).stat().st_size == 0
    ]
    inspection = inspect_model(checkpoint)
    missing.extend(item for item in inspection["missing"] if item != "tokenizer")
    return {"valid": not missing, "missing": missing, "path": str(checkpoint)}


def preflight(payload: dict[str, Any]) -> dict[str, Any]:
    """只读核验训练资源与基本数值参数，不加载权重或访问网络。

    Args:
        payload: model_path、dataset 版本清单和 training_args。
    Returns:
        valid、错误列表与当前模型检查结果。
    """
    errors = []
    model = None
    try:
        model_path = _local_path(payload["model_path"])
        model = inspect_model(model_path)
        if not model["complete"] or model["kind"] != "model":
            details = ", ".join(model["missing"]) or "所选资源是 adapter"
            errors.append(f"本地模型基座不完整：{details}")
    except (OSError, ValueError) as error:
        errors.append(f"模型路径不可读：{error}")
    try:
        dataset_path = _local_path(payload["dataset"]["path"])
        if not dataset_path.is_file():
            errors.append("数据集版本文件不存在")
        elif dataset_path.stat().st_size == 0:
            errors.append("数据集版本文件为空")
        else:
            with dataset_path.open("rb") as source:
                source.read(1)
    except (OSError, ValueError) as error:
        errors.append(f"数据集路径不可读：{error}")
    training_args = payload.get("training_args", {})
    for name in (
        "cutoff_len",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
    ):
        if name not in training_args:
            continue
        value = training_args[name]
        if name == "max_steps" and type(value) is int and value == -1:
            continue
        if type(value) is not int or value <= 0:
            errors.append(f"{name} 必须为正整数" + ("或 -1" if name == "max_steps" else ""))
    return {"valid": not errors, "errors": errors, "model": model}


def prepare_train(payload: dict[str, Any]) -> dict[str, Any]:
    """生成只引用本地资源的现有训练入口配置。

    Args:
        payload: job_id、model_path、dataset 清单、training_args 与可选续训路径。
    Returns:
        配置、输出目录、CLI 命令及必须应用的环境变量。
    """
    job_id = _identifier(payload["job_id"])
    model_path = _local_path(payload["model_path"])
    model = inspect_model(model_path)
    if not model["complete"] or model["kind"] != "model":
        raise ValueError("训练必须引用完整的本地模型基座")
    dataset = payload["dataset"]
    dataset_path = _local_path(dataset["path"])
    if not dataset_path.is_file():
        raise ValueError("数据集版本文件不存在")
    training_args = dict(payload.get("training_args", {}))
    forbidden = CONTROLLED_TRAINING_ARGUMENTS.intersection(training_args)
    if forbidden:
        raise ValueError(f"训练路径由服务端管理，不能设置：{', '.join(sorted(forbidden))}")
    if training_args.get("use_v1") or training_args.get("USE_V1"):
        raise ValueError("不支持新版训练入口 USE_V1")
    training_args.pop("use_v1", None)
    training_args.pop("USE_V1", None)
    job_root = ROOT / "jobs" / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    cache_dir = job_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    dataset_info_path = job_root / "dataset_info.json"
    output_dir = job_root / "output"
    entry = {"file_name": str(dataset_path)}
    for key in ("formatting", "columns", "tags", "ranking"):
        if key in dataset:
            entry[key] = dataset[key]
    _write_json(dataset_info_path, {"training_data": entry})
    configuration = {
        **training_args,
        "model_name_or_path": str(model_path),
        "dataset": "training_data",
        "dataset_dir": str(job_root),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "logging_dir": str(job_root / "logs"),
    }
    checkpoint = payload.get("resume_from_checkpoint")
    if checkpoint:
        inspection = check_checkpoint({"path": checkpoint})
        if not inspection["valid"]:
            raise ValueError(f"续训检查点不完整：{', '.join(inspection['missing'])}")
        configuration["resume_from_checkpoint"] = inspection["path"]
    # JSON 是合法 YAML，避免工具脚本再依赖一套序列化库。
    config_path = job_root / "train.yaml"
    _write_json(config_path, configuration)
    return {
        "config_path": str(config_path),
        "dataset_info_path": str(dataset_info_path),
        "output_dir": str(output_dir),
        "command": ["llamafactory-cli", "train", str(config_path)],
        "environment": {
            "USE_V1": "0",
            "USE_MODELSCOPE_HUB": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
    }


ACTIONS = {
    "init": initialize,
    "resources": resources,
    "catalog": catalog,
    "scan_models": scan_models,
    "model_download": model_download,
    "dataset_import": dataset_import,
    "dataset_preview": dataset_preview,
    "datasets_catalog": datasets_catalog,
    "prepare_train": prepare_train,
    "list_artifacts": list_artifacts,
    "read_metrics": read_metrics,
    "check_checkpoint": check_checkpoint,
    "preflight": preflight,
}


def main(argv: list[str] | None = None) -> int:
    """执行一个工具动作并输出机器可识别的最终结果。

    Args:
        argv: 动作名与 JSON 参数；省略时读取命令行。
    Returns:
        成功返回 0，失败返回 1。
    """
    arguments = sys.argv[1:] if argv is None else argv
    try:
        action, raw_payload = arguments
        payload = json.loads(raw_payload)
        if action not in ACTIONS or not isinstance(payload, dict):
            raise ValueError("未知动作或参数不是 JSON 对象")
        result = ACTIONS[action](payload)
        exit_code = 0
    except Exception as error:
        result = {"error": {"type": type(error).__name__, "message": str(error)}}
        exit_code = 1
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
