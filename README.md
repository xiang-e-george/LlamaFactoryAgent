# LlamaFactory 远端训练 API

在本机运行 FastAPI，通过已有远端 Docker Engine 创建任务容器，调用指定镜像中的 `llamafactory-cli train` 现有实现。远端无需部署 HTTP 服务、安装 SSH 组件或重新构建镜像；Windows 本机只需要 Python 与 Docker SDK，无需运行 Docker Desktop 或本地 Docker daemon。

- 模型仅从**魔搭社区 ModelScope** 下载，也可登记远端已有模型目录。
- 数据集仅支持**上传 JSON/JSONL 文件**或登记远端已有文件，不支持任何 Hub 数据集导入。
- 模型、数据与产物保存在远端；任务索引、结果和最终指标保存在本地 SQLite，完整运行日志归档到本地文件。
- 强制 `USE_V1=0`，不调用 LlamaFactory 新实现。

## 安装和启动

需要 Python 3.11 及以上版本。在项目目录执行 PowerShell：

```powershell
uv sync --extra dev
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
uv run python -m llamafactory_agent
```

已有虚拟环境时，也可用 `uv pip install -e ".[dev]"` 安装。激活环境后执行 `python -m llamafactory_agent` 或 `lfa-api` 启动。

默认地址为 `http://127.0.0.1:8765`，交互式接口文档位于 [Swagger UI](http://127.0.0.1:8765/docs)。API 默认仅监听本机，首版没有多用户鉴权。远端镜像需要包含现有 LlamaFactory CLI、ModelScope SDK 及可用的 NVIDIA 运行环境。

`.env.example` 已提供当前 Docker 连接和镜像 ID。主要配置如下：

| 配置 | 含义与默认值 |
|---|---|
| `LFA_DOCKER_CONTEXT` | 兼容已有配置，内容是 `tcp://地址:端口` 连接地址，不是 Docker context 名称 |
| `LFA_DOCKER_HOST` | 可选的连接地址配置，同时设置时优先于 `LFA_DOCKER_CONTEXT` |
| `LFA_DOCKER_IMAGE` | 指定已有镜像，默认 `2f51fed82caa`；运行时解析为完整 image ID |
| `LFA_STORAGE_ROOT` | 远端受管根目录，默认 `/tmp/llamafactory-agent` |
| `LFA_RUNTIME_DIR` | 本地状态和暂存上传目录，默认 `.runtime` |
| `LFA_MODEL_SCAN_DIRS` | 远端已有模型目录，默认空；支持逗号分隔或 JSON 数组 |
| `LFA_DATASET_SOURCE_DIRS` | 允许登记的数据目录，默认 `/opt/LlamaFactory/data`；支持逗号分隔或 JSON 数组 |
| `LFA_MAX_UPLOAD_BYTES` | 单个上传文件限制，默认 268435456 字节，即 256 MiB |
| `LFA_API_HOST` / `LFA_API_PORT` | 本地监听地址，默认 `127.0.0.1:8765` |
| `LFA_DOCKER_TIMEOUT` | Docker 请求超时，默认 30 秒 |
| `LFA_REMOTE_WORKER_TIMEOUT` | 同步探测容器的等待上限，默认 120 秒；不限制后台训练时长 |
| `LFA_POLL_INTERVAL` / `LFA_PROBE_TTL` | 调度间隔与资源探测缓存时间，默认分别为 2 秒、30 秒 |
| `LFA_CLEANUP_INTERVAL` | 终态任务归档与容器回收的对账间隔，默认 30 秒，最小 0.1 秒 |
| `LFA_FAILED_CONTAINER_RETENTION_HOURS` | 失败容器额外保留时长，默认 0 小时，即归档成功后立即回收；可设为 24，从 `finished_at` 起保留 24 小时 |

修改目录配置后重启本地 API。所有远端路径都指**远端 Linux 宿主机**；Windows 的 `D:\...` 路径不能作为远端挂载路径。

## 接口约定

创建下载、导入、训练及续训任务返回 HTTP `202`，任务标识在响应的 `id` 字段。查询 `/jobs/{id}` 获取 `status`、`result`、`error` 和容器 ID。完成的下载或导入任务，其 `result.id` 是之后提交训练需要的资源 ID。

任务响应同时提供归档和回收状态：

| 字段 | 含义 |
|---|---|
| `finished_at` | 任务进入终态的时间，也是失败容器保留期的起算时间 |
| `archived_at` | 日志及结果信息完成归档的时间 |
| `container_removed_at` | 任务容器确认已回收或已不存在的时间 |
| `cleanup_error` | 最近一次归档或回收错误；可恢复错误在后续对账中重试 |
| `cleanup_next_attempt_at` | 下一次归档或回收的尝试时间，包含失败重试或保留期结束时间 |
| `exit_code` | 已记录的容器退出码；未启动容器的任务可能为空 |
| `metrics` | 已保存的最终训练指标；没有训练指标时为空 |

`status` 表示任务执行结果，归档或回收失败不会将成功任务改为失败。尚未创建容器的任务无需回收；容器 ID 留在任务记录中用于追溯，不表示该容器仍然存在。

| 接口 | 用途 |
|---|---|
| `GET /health` | 仅检查本地 API，不访问远端 |
| `GET /resources` | 查询 GPU、显存、磁盘与镜像 |
| `GET /models?scope=local` | 查询已登记模型 |
| `GET /models?scope=catalog` | 读取指定镜像的魔搭模型候选及模板 |
| `POST /models/refresh` | 扫描受管目录和配置的已有模型目录 |
| `POST /models/downloads` | 从魔搭下载模型，参数为 `repo_id` 和可选 `revision` |
| `GET /datasets/catalog` | 读取允许目录下的 `dataset_info.json`，跳过在线来源 |
| `GET /datasets` | 查询已导入的数据集版本 |
| `POST /datasets/imports` | 用远端绝对路径登记 JSON/JSONL 文件 |
| `POST /datasets/uploads` | 用 multipart 上传 JSON/JSONL 文件 |
| `GET /datasets/{id}/preview` | 预览样本，可传 `version` 和 `limit`，最多 20 条 |
| `POST /training/preflight` | 校验模型、数据、训练参数，并查询当前 GPU |
| `POST /training/jobs` | 创建训练任务 |
| `GET /jobs` / `GET /jobs/{id}` | 查询任务列表或详情 |
| `GET /jobs/{id}/logs?tail=200` | 轮询日志末尾，首版不提供 SSE |
| `POST /jobs/{id}/cancel` | 取消排队任务，或请求停止运行中的容器 |
| `GET /training/jobs/{id}/artifacts` | 查询训练产物和检查点 |
| `GET /training/jobs/{id}/artifacts/download?path=相对路径` | 下载指定产物的 TAR 归档；路径须取自产物列表 |
| `POST /training/jobs/{id}/resume` | 从已结束任务的完整 `checkpoint-步数` 创建续训任务 |

创建类接口接受 `Idempotency-Key` 请求头。同一个键重试相同请求会返回原任务；将同一个键用于其他参数会返回 `409`。业务校验错误为 `400`，请求结构错误为 `422`，文件超过大小限制为 `413`，资源不存在为 `404`，远端操作不可用为 `503`。

## 远端资源与最小训练示例

先在 PowerShell 查询实际资源：

```powershell
$api = "http://127.0.0.1:8765"
Invoke-RestMethod "$api/resources"
Invoke-RestMethod "$api/models?scope=catalog"
Invoke-RestMethod "$api/datasets/catalog"
```

GPU 查询通过临时容器执行 `nvidia-smi`，因此查询本身需要远端 Docker 创建容器与使用 GPU 的能力。探测容器执行后会清理；下载、导入和训练任务容器在任务结束并成功归档后自动回收，历史日志仍可通过任务日志接口查询。

以下示例从魔搭下载 `Qwen/Qwen2.5-0.5B-Instruct`，登记远端的中文示例数据，再提交 3 步 SFT + LoRA 训练。运行前用数据集目录接口确认远端确实存在 `alpaca_zh_demo.json`；模型下载及训练都会占用远端存储和计算资源。

```powershell
$api = "http://127.0.0.1:8765"

function Wait-LfaJob {
    <#
    .SYNOPSIS
    等待后台任务完成，失败或取消时返回具体原因。
    .PARAMETER JobId
    创建接口返回的任务标识。
    .OUTPUTS
    成功任务的完整响应。
    #>
    param([string]$JobId)
    do {
        $job = Invoke-RestMethod "$api/jobs/$JobId"
        if ($job.status -in @("succeeded", "failed", "cancelled")) {
            break
        }
        Start-Sleep -Seconds 2
    } while ($true)
    if ($job.status -ne "succeeded") {
        throw "任务 $JobId 状态为 $($job.status)：$($job.error)"
    }
    return $job
}

$modelRequest = @{ repo_id = "Qwen/Qwen2.5-0.5B-Instruct" } | ConvertTo-Json
$modelJob = Invoke-RestMethod "$api/models/downloads" -Method Post `
    -ContentType "application/json" -Body $modelRequest `
    -Headers @{ "Idempotency-Key" = "qwen05-model-01" }
$modelJob = Wait-LfaJob $modelJob.id

$datasetRequest = @{
    source_path = "/opt/LlamaFactory/data/alpaca_zh_demo.json"
    formatting = "alpaca"
} | ConvertTo-Json
$datasetJob = Invoke-RestMethod "$api/datasets/imports" -Method Post `
    -ContentType "application/json" -Body $datasetRequest `
    -Headers @{ "Idempotency-Key" = "alpaca-zh-data-01" }
$datasetJob = Wait-LfaJob $datasetJob.id

$trainingRequest = @{
    model_id = $modelJob.result.id
    dataset_id = $datasetJob.result.id
    dataset_version = $datasetJob.result.version
    training_args = @{
        stage = "sft"
        template = "qwen"
        finetuning_type = "lora"
        max_steps = 3
        cutoff_len = 256
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 1
        logging_steps = 1
        save_steps = 1
        save_only_model = $false
        bf16 = $true
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod "$api/training/preflight" -Method Post `
    -ContentType "application/json" -Body $trainingRequest
$trainingJob = Invoke-RestMethod "$api/training/jobs" -Method Post `
    -ContentType "application/json" -Body $trainingRequest `
    -Headers @{ "Idempotency-Key" = "qwen05-sft-01" }
$trainingJob.id
Invoke-RestMethod "$api/jobs/$($trainingJob.id)/logs?tail=200"
```

`bf16` 示例适用于当前计划使用的 RTX 4080；更换硬件时需要按硬件能力设置精度。模型候选只表示镜像中有对应预设，不等于该模型已经下载、仓库允许访问或在当前 GPU 上可训练。预检返回的是参数检查和当前显存快照，不保证任何配置都不会显存不足。

上传本地文本数据可直接在 `/docs` 的 `/datasets/uploads` 接口选择文件；PowerShell 7 也支持：

```powershell
$api = "http://127.0.0.1:8765"
$datasetFile = Get-Item -LiteralPath "./opensource/LlamaFactory/data/alpaca_zh_demo.json"
$uploadJob = Invoke-RestMethod "$api/datasets/uploads" -Method Post -Form @{
    file = $datasetFile
    formatting = "alpaca"
    columns = '{"prompt":"instruction","query":"input","response":"output"}'
    ranking = "false"
}
$uploadJob.id
```

`columns` 与 `tags` 在上传表单中为 JSON 字符串；远端文件登记接口中为 JSON 对象。对话数据可使用 `formatting="sharegpt"` 并设置角色标签映射；偏好数据需要 `ranking=true`。服务对导入数据检查字段和样本结构，成功后产生独立资源版本。训练引用资源版本，不直接引用用户提交的文件路径。

## 运行和存储规则

- 同一运行目录只允许一个本地调度器。训练队列同时执行一个任务，下载和导入共用另一个串行队列；不要通过多个 Uvicorn worker 扩容。
- 关闭本地 API 不会停止已在远端启动的任务；排队任务等本地 API 恢复后再派发。离线期间容器保留，恢复后根据 SQLite 记录和容器标签补做归档及回收；网络中断不会直接判定训练失败。
- 取消运行任务会停止对应容器，**不保证取消瞬间保存检查点**。续训从已经完整写出的检查点创建新任务，需要权重、优化器、调度器与训练状态；最终 adapter 文件不能替代完整检查点。
- 训练配置中的模型路径、数据集路径、输出目录、日志路径和缓存由服务管理。模型与数据挂载只读，当前任务目录可写；训练缓存放在当前任务目录，魔搭下载缓存使用受管共享目录。
- 远端暂定模型、数据、下载缓存、任务和传入文件分别位于受管根目录下的 `models/`、`datasets/`、`cache/`、`jobs/`、`incoming/`。`/tmp` 可能被系统清理，仅适合当前验证阶段。
- 本地 `.runtime/control.sqlite3` 与远端文件共同构成任务状态。不要在任务运行中删除 SQLite 或手工移除任务容器。模型、数据集、训练产物与检查点不会随任务容器自动删除。
- 修改根目录不会自动迁移既有模型与任务；正式目录规范和迁移应单独安排。

### 任务归档与容器回收

成功、失败和取消均为终态。服务先把容器完整日志保存到本地 `.runtime/jobs/{job_id}/console.log`，把任务结果、最终 `metrics` 与 `exit_code` 保存到 SQLite，再记录 `archived_at`。本地目录随 `LFA_RUNTIME_DIR` 配置变化。

默认所有终态任务在归档成功后立即尝试回收容器，不额外保留失败容器。需要检查失败现场时，将 `LFA_FAILED_CONTAINER_RETENTION_HOURS` 设为 `24`；保留期从 `finished_at` 起算，到期后仍须先完成归档再回收。没有创建容器的任务无需执行容器回收。

归档或删除失败时记录 `cleanup_error`，保留任务执行结果，并按 `LFA_CLEANUP_INTERVAL` 重试，默认每 30 秒对账一次。本地 API 离线时不会触发清理，恢复后会继续处理未完成的归档与回收。

若容器在归档完成前已被外部移除，缺失内容可能无法恢复。服务记录 `container_removed_at` 和明确的 `cleanup_error`，保持 `archived_at` 为空，不再反复重试已不存在的容器；这不表示归档成功，也不会删除已有日志或远端资源。

归档后的日志接口读取本地日志，不依赖原任务容器。训练产物和检查点继续保留在远端宿主机；产物查询、下载和续训也不依赖原容器。下载通过一次性只读容器读取远端文件并输出 TAR 归档，传输结束后清理该临时容器；若本地进程中断留下读取容器，API 重启后的对账会补做清理。整个流程仍只使用 Docker API，不需要 SSH 或远端常驻服务。

## 本地验证

```powershell
uv run pytest -q
uv run ruff check src tests
```

本地测试使用替代 Docker 服务和临时文件验证 API、数据校验、容器配置与任务状态机。真实镜像兼容性、魔搭网络和 GPU 训练需另行在目标远端验收。
