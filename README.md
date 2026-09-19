# Qichi Engine

一个**单用户 QQ 情感陪伴机器人**的引擎层。

设计原则：**角色由主模型决定，代码只负责可靠传输、真实上下文、可追溯记忆和能力边界。**
代码不做关键词情绪分类、不做逐句正则剥离、不做多级固定文案兜底，也不替角色决定说什么。

> 这个仓库**只开源引擎**。角色设定、真实对话、记忆库、部署记录、密钥都不在仓库内。
> 发布本仓库时请从**快照**建立，不要携带私有开发历史。

## 仓库里有什么

| 目录 | 内容 |
| --- | --- |
| `src/qichi/` | 引擎源码：对话热路径、上下文装配、记忆层、QQ 通道、语音编排、面板后端 |
| `tests/` | 测试套件（1600+ 条） |
| `scripts/` | 运维与诊断脚本（preflight、记忆修复、A/B 取样等） |
| `dashboard/` | 观测面板前端（静态资源） |
| `migrations/` | SQLite 迁移 |
| `config.example.yaml` | 配置示例 |
| `doc/运行时角色核心.example.md` | **通用占位角色**（不是可用人设，见下） |

## 仓库不包含什么（你必须自己准备）

引擎运行需要三样东西，它们**故意不入库**。缺任何一样都不会「凑合跑」，而是**失败关闭**。

### 1. 角色核心（`persona.system_prompt_file`）

整条对话热路径**唯一注入的角色文本**。引擎自己不带人设：它只加载配置指定的那一个文件，
同目录的其它文档永远不会被带进上下文。

- 仓库只带一份通用示例 `doc/运行时角色核心.example.md`。**复制成你自己的文件再改写**，例如
  `doc/我的角色.md`，然后把 `config.example.yaml` 的 `persona.system_prompt_file` 指过去。
- 约束：按项目根目录解析、**必须落在项目根目录内**、UTF-8、非空、不超过 32768 字节。
- 缺失或非法时，装配阶段直接抛 `RuntimeAssemblyError: runtime role prompt is unavailable`——
  **不会退回任何内置人设**。

### 2. Tokenizer 产物 `runtime/model-cache/v4-tokenizer.json`

`runtime/` 在 `.gitignore` 里，所以仓库不带；但 `preflight`、生产装配和若干脚本都要用它。

下载地址与哈希**已经钉在** `src/qichi/dialogue/model_capability.py` 的 `MODEL_MANIFESTS` 里，
加载时会按 manifest 里的 `tokenizer_sha256` 校验，不匹配就拒绝：

| 模型 | tokenizer.json |
| --- | --- |
| `deepseek-v4-flash` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json |
| `deepseek-v4-pro` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json |
| `deepseek-ai/DeepSeek-V3.2` | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/tokenizer.json |

flash 与 pro 的 `tokenizer.json` 逐字节相同（SHA256 都是
`8F9F37CA37FDC4F5FD36D5CF4D3B0E8392EDB4E894FD10CC0D70B4957C8633CF`），所以同一份产物两边通用。

```powershell
New-Item -ItemType Directory -Force runtime\model-cache | Out-Null
Invoke-WebRequest -Uri <上表对应 URL> -OutFile runtime\model-cache\v4-tokenizer.json
(Get-FileHash runtime\model-cache\v4-tokenizer.json -Algorithm SHA256).Hash
```

缺失时：`preflight.py` 输出 `BLOCKED: tokenizer artifact: tokenizer artifact is unavailable`。

### 3. 提供方能力证据 `runtime/<主模型名>-capability.json`

例如 `runtime/deepseek-v4-flash-capability.json`（也可用 `preflight.py --evidence <路径>` 显式指定）。

**为什么别人不能替你生成**：它是「某一时刻、某一家供应商，对某个确切模型实测到的总窗口」，
是有时效、有出处的观察。引擎刻意把「模型架构上限」和「供应商实际承诺」分开，
不会拿架构文档里的数字冒充供应商承诺。

格式**恰好**是这 7 个键，多一个少一个都会被拒绝：

```json
{
  "schema": "qichi.provider-capability/v1",
  "provider": "deepseek",
  "model_id": "deepseek-v4-flash",
  "context_tokens": 1048576,
  "source": "https://api-docs.deepseek.com/quick_start/pricing",
  "retrieved_at_utc": "2026-09-13T00:00:00+00:00",
  "observation": "deepseek-v4-flash: provider documents a 1048576-token total context window"
}
```

校验规则：`schema` 精确匹配；`provider`/`model_id` 必须与配置一致；`source` 必须以 `http://`
或 `https://` 开头；`retrieved_at_utc` 必须是**带时区**的 ISO 时间；`observation` 必须包含
`model_id`；`context_tokens` 必须是正整数。

缺失时：`preflight.py` 输出 `BLOCKED: provider capability evidence: ModelCapabilityError`，
生产启动拒绝。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"

# 1) 角色核心：复制示例并改成你自己的，然后改 config.example.yaml 指向它
Copy-Item doc\运行时角色核心.example.md doc\我的角色.md

# 2) tokenizer（见上表）
# 3) 能力证据（见上）

.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\preflight.py --config config.example.yaml
```

密钥**只从环境变量读取**，不写入仓库、配置示例、日志或文档：

| 变量 | 用途 |
| --- | --- |
| `QICHI_OWNER_QQ` | 唯一对话对象的 QQ 号 |
| `NAPCAT_WS_URL` / `NAPCAT_HTTP_URL` / `NAPCAT_ACCESS_TOKEN` | OneBot/QQ 通道 |
| `DEEPSEEK_API_KEY` | 主模型 |
| `TAVILY_API_KEY` | 文本联网（可选） |
| `SERPAPI_API_KEY` | 以图搜图（可选） |
| `DASHSCOPE_API_KEY` | 语音合成（可选） |

实际收发 QQ 消息需要外部 OneBot 实现（本项目按 NapCat 部署，但不是唯一选择）。

## 先看一眼它怎么跑（不需要 QQ、密钥或联网）

```powershell
.venv\Scripts\python scripts\demo_offline.py
```

它跑的是**真正的**引擎链路——真实的事件持久化、真实的上下文装配、真实的发送与 outbox——
只把模型和 QQ 通道换成假替身。每一轮都会打印引擎**真正装配出来的上下文**：

```
角色核心 → 能力与边界事实 → 历史原文证据 → 本轮事实 → 当前输入
```

所以它说明的是「引擎把什么喂给模型、又把什么发了出去」，不是「模型会说什么」。
后者需要你自己的密钥和角色核心。

## 测试

```powershell
.venv\Scripts\python -m pytest -q
```

当前结果：**1622 passed, 1 skipped**。

跳过的那条是
`tests/test_preflight.py::test_preflight_is_read_only_and_reports_production_assembly`——
它检验「**生产装配是否就绪**」，需要上面第 2、3 件产物。缺件时它**跳过，而不是假装通过**：
没有 tokenizer 与供应商能力证据，就绪与否根本无从判定。补齐产物后它会真正执行。

CI 见 `.github/workflows/ci.yml`：Windows + Python 3.11/3.14 上跑完整测试。
（暂不加 Linux：这套引擎是 Windows 部署的，测试里也有 Windows 专有路径，
加 Linux 需要一轮真机验证，不能靠推测。）

## 关于注释里的历史依据

引擎源码和测试的注释保留了当时的工程依据，并用「历史诊断 §3」这类**中性标签**指代本项目的
**私有**设计与诊断记录。这些记录不在仓库内，标签不对应任何仓库内文件——只影响可读性，
不影响任何行为。
