# Qichi Engine

一个**单用户 QQ 情感陪伴机器人**的引擎层。

设计原则：**角色由主模型决定，代码只负责可靠传输、真实上下文、可追溯记忆和能力边界。**
代码不做关键词情绪分类、不做逐句正则剥离、不做多级固定文案兜底，也不替角色决定说什么。

> 这个仓库**只开源引擎**。角色设定、真实对话、记忆库、部署记录、密钥都不在仓库内。
> 发布本仓库时请从**快照**建立，不要携带私有开发历史。

**实现链路、上下文装配、记忆层与设计取舍见 [`doc/架构说明.md`](doc/架构说明.md)。**

---

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/qichi/` | 引擎源码：对话热路径、上下文装配、记忆层、QQ 通道、语音编排、面板后端 |
| `tests/` | 测试套件（1600+ 条） |
| `scripts/` | 入口与运维脚本（preflight、启动栈、面板、离线演示、记忆修复等） |
| `dashboard/` | 只读观测面板前端（静态资源） |
| `migrations/` | SQLite 迁移 |
| `config.example.yaml` | 配置示例 |
| `doc/运行时角色核心.example.md` | **通用占位角色**（不是可用人设） |
| `doc/架构说明.md` | 架构与链路说明 |

---

## 一、跑起来需要你补全的东西

引擎**故意不带**下面这些东西。缺任何一样都不会凑合跑，而是**失败关闭**——宁可起不来，也不要一条会静默失败的链路。

| # | 要补的东西 | 必需性 | 怎么补 |
| --- | --- | --- | --- |
| 1 | **你自己的角色核心** `doc/你的角色.md` | 必需 | 复制 `doc/运行时角色核心.example.md` 再改写；见第二节 |
| 2 | **配置文件** `config.yaml` | 必需 | 复制 `config.example.yaml`，改 `persona.system_prompt_file` 等 |
| 3 | **环境变量（密钥）** | 必需 | 见第五节 |
| 4 | **tokenizer 产物** `runtime/model-cache/v4-tokenizer.json` | 生产装配必需 | 按 manifest 钉住的 URL 下载；见 1.1 |
| 5 | **供应商能力证据** `runtime/<主模型名>-capability.json` | 生产装配必需 | 自己实测后写一份；见 1.2 |
| 6 | **QQ 原生表情目录** `data/qq-expression-catalog.json` | 只有要开表情才需要 | 参考 `data/qq-expression-catalog.example.json`；见 1.3 |
| 7 | NapCat / QQ 路径 | 只有跑真机才需要 | 环境变量 `QICHI_NAPCAT_ROOT`（必须）、`QICHI_QQ_EXE`（可选） |

> `data/qichi.sqlite3`、`data/voice/`、`runtime/*.log` 等运行期文件会自己生成，不需要你准备。

### 1.1 tokenizer 产物

`runtime/` 在 `.gitignore` 里，仓库不带这份产物；但 `preflight`、生产装配和若干脚本都要用它。
下载地址与哈希**已钉在** `src/qichi/dialogue/model_capability.py` 的 `MODEL_MANIFESTS` 里，
加载时按 `tokenizer_sha256` 校验，不匹配就拒绝。

| 模型 | tokenizer.json |
| --- | --- |
| `deepseek-v4-flash` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json |
| `deepseek-v4-pro` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json |
| `deepseek-ai/DeepSeek-V3.2` | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/tokenizer.json |

flash 与 pro 的 `tokenizer.json` **逐字节相同**（SHA256 都是
`8F9F37CA37FDC4F5FD36D5CF4D3B0E8392EDB4E894FD10CC0D70B4957C8633CF`），同一份产物两边通用。

```powershell
New-Item -ItemType Directory -Force runtime\model-cache | Out-Null
Invoke-WebRequest -Uri <上表对应 URL> -OutFile runtime\model-cache\v4-tokenizer.json
(Get-FileHash runtime\model-cache\v4-tokenizer.json -Algorithm SHA256).Hash   # 应等于上面那串
```

缺失时：`preflight.py` 输出 `BLOCKED: tokenizer artifact: tokenizer artifact is unavailable`。

### 1.2 供应商能力证据

例如 `runtime/deepseek-v4-flash-capability.json`（也可用 `preflight.py --evidence <路径>` 显式指定）。

**为什么别人不能替你生成**：它是「某一时刻、某一家供应商，对某个确切模型实测到的总窗口」，
是有时效、有出处的观察。引擎刻意把「模型架构上限」和「供应商实际承诺」分开，
**绝不拿架构文档里的数字冒充供应商承诺**。

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

校验规则：`schema` 精确匹配；`provider`/`model_id` 必须与配置一致；`source` 必须以 `http://` 或
`https://` 开头；`retrieved_at_utc` 必须是**带时区**的 ISO 时间；`observation` 必须包含
`model_id`；`context_tokens` 必须是正整数。

缺失时：`preflight.py` 输出 `BLOCKED: provider capability evidence: ModelCapabilityError`，生产启动拒绝。

### 1.3 QQ 原生表情目录（可选）

要开 `expression.qq_face`，需要一个**可核验**的语义目录：face 名 → QQ 表情 id，另带来源与哈希
（`source` / `source_commit` / `source_hash`）。仓库只带一份**示例**
`data/qq-expression-catalog.example.json`，而 `config.example.yaml` 里的 `runtime_catalog`
指的是 `data/qq-expression-catalog.json`（**由你提供**）。

`source_hash` = 除 `source_hash` 外全部字段的规范 JSON
（`ensure_ascii=False, sort_keys=True, separators=(",", ":")`）的 SHA256。

---

## 二、人设怎么设定

### 2.1 只有一处来源

**整条对话热路径唯一注入的角色文本**就是 `persona.system_prompt_file` 指向的那一个文件。
引擎自己不带人设，也**永远不会**把别的文档拼进去。所以：

- 复制 `doc/运行时角色核心.example.md` 成你自己的文件（例如 `doc/我的角色.md`），改写它；
- 把 `config.example.yaml` 的 `persona.system_prompt_file` 指过去。

约束（`src/qichi/dialogue/prompt_loader.py`）：

| 约束 | 值 | 违反后 |
| --- | --- | --- |
| 路径 | 按项目根解析，**必须落在项目根目录内** | `runtime prompt must stay inside the project root` |
| 编码 | UTF-8 | `runtime prompt must be UTF-8` |
| 大小 | ≤ 32768 字节 | `runtime prompt exceeds 32768 bytes` |
| 内容 | 非空 | `runtime prompt is empty` |

任何一条不满足，装配阶段直接抛 `RuntimeAssemblyError: runtime role prompt is unavailable`——
**不会退回任何内置人设**。这是故意的：宁可起不来，也不要悄悄换一个人。

### 2.2 薄人设该怎么写

这份文本**每一轮都会注入**，是每字节成本最高的文本，也是她是谁的唯一来源。建议只写四类：

1. **身份与语气的一句话**：她是谁、说话什么路子。
2. **事实纪律**：只把当前消息、明确给出的历史原文、带来源的记忆和真实工具结果当事实；
   不确定就说不确定；不补证据没有给出的数量、频率、程度或细节。
3. **能力边界**：没有现实身体、视觉或外部行动能力；被问到时如实说明。
4. **表达自由度**：可以自然表达喜欢、想念、犹豫、害羞、拒绝；先给出能推进对话的具体答案。

**不要写**（写了就会每轮重复，或把代码该管的事塞进提示词）：

- 逐条关键词规则、如果你说 X 就回 Y——那是把理解权从模型手里拿走；
- 情绪分类表、固定话术池、口癖清单；
- 与某个人有关的具体事实（职业、作息、约定……）——**那属于记忆与关系状态，不属于人设**。
  人设回答我是谁，记忆回答我们之间发生过什么；两者混在一起，人设就会随事实一起漂移；
- 长篇设定文档：注入成本高、边际收益低，还会让模型把朗读设定当成任务。

### 2.3 人设一致性怎么保证

- **单一来源**：角色文本只有一处，不存在人设 A 用于回复、人设 B 用于主动消息。
- **同一入口**：主动消息复用**同一角色提示、同一上下文、同一生成入口**（`use_same_dialogue_engine: true`），
  不走独立话术池、不走独立人格。
- **代码只给事实**：当前时间、消息是谁发的、引用的是哪一条、哪些工具真的可用、哪些能力**明确不可用**，
  都由代码保证并标注来源；性格、情绪、亲密程度、接话方式、是否引用某条消息由模型决定
  （`decision_authority` 一节把这条界线写进了配置）。
- **事实与角色分离**：用户事实必须有原话证据，角色自己的旧话不能当作用户事实
  （`distinguish_user_fact_from_assistant_history`）；过去的想象不等于本轮同意。

---

## 三、外部功能怎么开关，哪些需要额外 API

所有开关都在 `config.yaml`。**关着 = 不建客户端、不声明能力、行为与没有这个功能完全一致**；
**开着但缺 key = 拒绝启动**（失败关闭，避免静默失败的能力声明）。

| 功能 | 配置键 | 示例默认 | 额外需要 | 说明 |
| --- | --- | --- | --- | --- |
| 主模型对话 | `llm.provider` / `llm.primary.model` | deepseek | `DEEPSEEK_API_KEY` | 热路径每轮**只调用一次** |
| 记忆抽取（后台） | `llm.background_model` / `memory.*` | 同主模型 | 同上 | 跑在**冻结会话**上，不占用户等待时间 |
| QQ 通道 | `transport.*` | napcat_onebot11 | NapCat（外部程序）+ `NAPCAT_WS_URL` `NAPCAT_HTTP_URL` `NAPCAT_ACCESS_TOKEN` | 只用 OneBot 11 协议，不绑定 NapCat |
| 文字联网检索 | `net.enabled` | **true** | `TAVILY_API_KEY` | Tavily；默认不主动查、带图轮也不查 |
| 以图搜图 | `net.image_api_key_env` | — | `SERPAPI_API_KEY` | **图只经过这一家**，文本检索那家看不到照片；先问再查 |
| 识图（看图） | `vision.enabled` | true | 无（用主模型/视觉档） | 关闭即回到只标记收到图片、不含视觉结果 |
| 语音合成 TTS | `voice.enabled` | **true** | `DASHSCOPE_API_KEY` + **你自己的 voice_id** | 见下 |
| QQ 原生表情 | `expression.qq_face.enabled` | false | `data/qq-expression-catalog.json` | 缺目录启动拒绝 |
| 消息回应 reaction | `expression.message_reaction.enabled` | false | 回应目录（仓库不携带） | 保持关闭 |
| Unicode emoji | `expression.unicode_emoji.enabled` | true | 无 | 由模型在同一次生成里选 |
| 主动消息 | `initiative.enabled` | true | 无 | 节流 + 静默时段 + 模型可跳过 |
| 观测面板 | 独立进程 | — | 无 | `scripts/dashboard_server.py`，只读 |

> **注意**：示例配置里 `net.enabled` 与 `voice.enabled` 是 **true**（那是生产快照）。
> 如果你没有 Tavily / DashScope 的 key，请把对应 `enabled` 改成 `false`，否则启动会被拒绝——
> 这正是设计意图：说得出能力就必须做得出来。

**语音的 voice_id 必须换成你自己的**：`config.example.yaml` 里写的是占位符
`REPLACE_WITH_YOUR_OWN_VOICE_ID`。音色由你在供应商侧设计/选定一次，然后固定写进配置，
运行时不再重新设计（避免每次发起都变成掷骰子）。`voice.model` 必须与设计该音色时用的模型一致。

---

## 四、快速开始

### 前置条件

- **一个空闲的 QQ 号**（机器人账号）
- Windows（生产启动脚本是 Windows 专用；引擎代码本身跨平台）
- Python ≥ 3.11
- NapCat 或其它 OneBot 11 实现（只有跑真机才需要）

### 步骤

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"

# 1) 角色核心：复制示例并改成你自己的，然后改 config.yaml 指向它
Copy-Item doc\运行时角色核心.example.md doc\我的角色.md
Copy-Item config.example.yaml config.yaml

# 2) tokenizer（见 1.1）   3) 能力证据（见 1.2）   4) 环境变量（见第五节）

# 先看一眼它怎么跑（不需要 QQ、密钥或联网）
.venv\Scripts\python scripts\demo_offline.py

# 只读体检：不联网、不发消息、不写 READY
.venv\Scripts\python scripts\preflight.py --config config.yaml

# 跑测试
.venv\Scripts\python -m pytest -q
```

跑真机（Windows，需要先装好 NapCat 并登录 QQ）：

```powershell
setx QICHI_NAPCAT_ROOT "D:\NapCatQQ"     # 你的 NapCat 根目录
setx QICHI_QQ_EXE "C:\...\QQ.exe"       # 可选；不设则用 <NapCatRoot>\qq\QQ.exe
.\start_bot.bat                           # 启动生产栈
.\stop_bot.bat                            # 按证据停止（不裸 taskkill）
.venv\Scripts\python scripts\check_ready.py runtime\qichi-ready.json runtime\qichi.lock --owner <你的QQ> --bot <机器人QQ>
```

`start_bot.bat` **不含任何机器专属路径**：`QICHI_NAPCAT_ROOT` 没设就报错退出，不会猜。

---

## 五、环境变量（密钥）

密钥**只从环境变量读取**，不写入仓库、配置示例、日志或文档。Windows 上推荐 `setx`（写用户环境）。

| 变量 | 用途 | 必需性 |
| --- | --- | --- |
| `QICHI_OWNER_QQ` | 唯一对话对象的 QQ 号 | 必需 |
| `DEEPSEEK_API_KEY` | 主模型 | 必需 |
| `NAPCAT_WS_URL` | OneBot 正向 WebSocket | 跑真机必需 |
| `NAPCAT_HTTP_URL` | OneBot HTTP | 跑真机必需 |
| `NAPCAT_ACCESS_TOKEN` | OneBot 令牌 | 跑真机必需 |
| `DASHSCOPE_API_KEY` | 语音合成（百炼） | `voice.enabled: true` 时必需 |
| `TAVILY_API_KEY` | 文字联网检索 | `net.enabled: true` 时必需 |
| `SERPAPI_API_KEY` | 以图搜图 | `net.enabled: true` 时必需 |
| `QICHI_NAPCAT_ROOT` | NapCat 根目录 | 跑真机必需 |
| `QICHI_QQ_EXE` | QQ 可执行文件 | 可选 |

日志会做密钥脱敏（`observability.redact_secrets`），轮次追踪（`turn_trace_events`）
**只记结构事实、不记对话原文**。

---

## 六、离线演示（零成本、无网络）

```powershell
.venv\Scripts\python scripts\demo_offline.py
```

它跑的是**真正的**引擎链路——真实的事件持久化、真实的上下文装配、真实的发送与 outbox——
只把模型和 QQ 通道换成假替身。每一轮都会打印引擎**真正装配出来的上下文**：

```
角色核心 → 能力与边界事实 → 历史原文证据 → 本轮事实 → 当前输入
```

所以它说明的是「引擎把什么喂给模型、又把什么发了出去」，不是「模型会说什么」。
后者需要你自己的密钥和角色核心。加 `--db demo.sqlite3` 可以保留数据库自己查。

---

## 七、测试与 CI

```powershell
.venv\Scripts\python -m pytest -q
```

当前结果：**1622 passed, 1 skipped**。

跳过的那条是 `tests/test_preflight.py::test_preflight_is_read_only_and_reports_production_assembly`——
它检验**生产装配是否就绪**，需要 1.1/1.2 两件产物。缺件时它**跳过，而不是假装通过**：
没有 tokenizer 与供应商能力证据，就绪与否根本无从判定。补齐产物后它会真正执行。

CI 见 `.github/workflows/ci.yml`：Windows + Python 3.11/3.14 上跑完整测试。
（暂不加 Linux：这套引擎是 Windows 部署的，测试里也有 Windows 专有路径，
加 Linux 需要一轮真机验证，不能靠推测。）

---

## 八、生产运行与观测

| 命令 | 作用 |
| --- | --- |
| `start_bot.bat` | 启动生产栈（隐藏窗口；日志 `runtime\start-production.log`） |
| `stop_bot.bat` → `scripts/stop_production.ps1` | 按证据停止（只停属于本项目的进程，**不裸 taskkill**） |
| `scripts/check_ready.py` | 校验 READY marker 与实例锁身份 |
| `scripts/preflight.py` | 只读体检：不联网、不发消息、不写 READY |
| `scripts/dashboard_server.py` | 只读观测面板（回环地址，默认 8765） |
| `scripts/start_stack.py` | 启动栈本体：正向 WS、SQLite 恢复、READY marker |

---

## 九、密钥纪律（请遵守）

- 只从环境变量读密钥；**不要**写进配置、日志、文档或提交。
- `config.example.yaml` 里没有任何真实密钥；`api_key_env` 只是**变量名**。
- 如果某个 key 曾经进过版本历史，请先去供应商轮换它，再发布。
- `observability.redact_secrets: true` 会脱敏日志，但这不构成随手写 key 的理由。

---

## 十、关于注释里的历史依据

引擎源码和测试的注释保留了当时的工程依据，并用「历史诊断 §3」这类**中性标签**指代本项目的
**私有**设计与诊断记录。这些记录不在仓库内，标签不对应任何仓库内文件——只影响可读性，
不影响任何行为。
