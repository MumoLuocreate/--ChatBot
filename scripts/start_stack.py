"""Start the fail-closed production Qichi runtime.

The command stays attached to the runtime until Ctrl+C.  It performs no
semantic fallback: provider, NapCat identity, SQLite recovery and the unique
Forward WS must all pass before the READY marker is written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import os
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.config import ConfigError, load_config
from qichi.readiness import ReadinessError, reclaim_stale_runtime_artifacts, runtime_process_probe
from qichi.runtime import build_production_runtime
from qichi.transport.onebot_client import OneBotClient


def _hydrate_user_environment() -> None:
    """Make an already-open PowerShell see the user's persisted env values."""
    if os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in (
                "QICHI_OWNER_QQ",
                "NAPCAT_WS_URL",
                "NAPCAT_HTTP_URL",
                "NAPCAT_ACCESS_TOKEN",
                "SILICONFLOW_API_KEY",
                "DEEPSEEK_API_KEY",
                # 联网（2026-09-13）：文本检索与图搜各一个 key。用户用 setx 写进用户环境后，
                # 这里读注册表就能拿到——经 WMI 拉起的进程不会自动继承新 setx 的变量。
                "TAVILY_API_KEY",
                "SERPAPI_API_KEY",
            ):
                if name in os.environ:
                    continue
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if isinstance(value, str) and value:
                    os.environ[name] = value
    except (ImportError, OSError):
        return



def _record_effective_features(config: object, project_root: Path) -> None:
    """Publish the switches this process actually loaded, for the read-only panel.

    A config file edited after startup is a request, not a fact, so the panel must read
    what is running rather than what the file currently says.  Failures here must never
    keep the bot from starting: the panel is an enhancement.
    """
    try:
        storage = getattr(config, "storage", None)
        expression = getattr(config, "expression", None)
        memory = getattr(config, "memory", None)
        initiative = getattr(config, "initiative", None)
        raw_path = str(getattr(storage, "database_path", ""))
        if not raw_path:
            return
        database_path = Path(raw_path)
        if not database_path.is_absolute():
            database_path = project_root / database_path
        emoji = getattr(expression, "unicode_emoji", None)
        face = getattr(expression, "qq_face", None)
        reaction = getattr(expression, "message_reaction", None)
        sticker = getattr(expression, "custom_sticker", None)
        vision = getattr(config, "vision", None)
        net = getattr(config, "net", None)
        features = {
            "initiative_enabled": bool(getattr(initiative, "enabled", False)),
            # ExpressionSettings has no top-level switch: each channel has its own.
            "unicode_emoji_enabled": bool(getattr(emoji, "enabled", False)),
            "custom_sticker_enabled": bool(getattr(sticker, "enabled", False)),
            "qq_face_enabled": bool(getattr(face, "enabled", False)),
            "message_reaction_enabled": bool(getattr(reaction, "enabled", False)),
            "memory_auto_commit": str(getattr(memory, "auto_commit", "")),
            # What this process would actually do, not what the provider can do:
            # the switch is the gate, and it is read from the loaded config.
            "vision_available": bool(getattr(vision, "enabled", False)),
            # 联网（2026-09-14）：开关是门；客户端建得起来还要看 key 在不在。
            # 这里只报**名字在不在**，绝不读值、更不写日志。
            "external_tools_available": bool(getattr(net, "enabled", False)),
            "external_search_ready": bool(
                os.environ.get(str(getattr(net, "api_key_env", "")), "").strip()
            ),
            "external_image_search_ready": bool(
                os.environ.get(str(getattr(net, "image_api_key_env", "")), "").strip()
            ),
        }
        connection = sqlite3.connect(database_path, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute(
                "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, "
                "updated_at_utc = excluded.updated_at_utc",
                ("runtime:features", json.dumps(features, ensure_ascii=False, sort_keys=True),
                 datetime.now(timezone.utc).isoformat()),
            )
        finally:
            connection.close()
    except Exception:
        pass


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_ROOT / "config.example.yaml")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
    )
    parser.add_argument("--ready", type=Path, default=_ROOT / "runtime" / "qichi-ready.json")
    parser.add_argument("--lock", type=Path, default=_ROOT / "runtime" / "qichi.lock")
    parser.add_argument(
        "--exit-when-ready",
        action="store_true",
        help="write READY and stop immediately; intended only for lifecycle tests",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    _hydrate_user_environment()
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"start_stack: config failed ({type(error).__name__})", file=sys.stderr)
        return 2

    try:
        archived = reclaim_stale_runtime_artifacts(
            args.ready,
            args.lock,
            runtime_probe=lambda pid, started: runtime_process_probe(
                pid,
                acquired_at_utc=started,
                expected_executable=sys.executable,
            ),
        )
    except ReadinessError as error:
        print(f"start_stack: recovery failed ({type(error).__name__})", file=sys.stderr)
        return 1
    if archived:
        print(f"start_stack: archived {len(archived)} stale runtime artifact(s)", flush=True)

    client = OneBotClient(
        config.transport.http_url,
        config.transport.websocket_url,
        config.transport.access_token,
        timeout=max(10.0, float(config.llm.primary.timeout_seconds)),
    )
    runtime = None
    try:
        login = await client.get_login_info()
        bot_qq = login.get("user_id") if hasattr(login, "get") else None
        if isinstance(bot_qq, bool) or not isinstance(bot_qq, (int, str)) or not str(bot_qq).isdecimal():
            raise RuntimeError("NapCat get_login_info returned an invalid identity")
        runtime = build_production_runtime(
            config,
            project_root=_ROOT,
            bot_qq=str(bot_qq),
            onebot_client=client,
            evidence_path=args.evidence,
            marker_path=args.ready,
            lock_path=args.lock,
        )
        await runtime.start()
        _record_effective_features(config, _ROOT)
        provider = getattr(config.llm, "provider", "configured-provider")
        model = getattr(config.llm.primary, "model", "configured-model")
        print(f"start_stack: READY (NapCat, SQLite, {provider}/{model}, workers)", flush=True)
        if args.exit_when_ready:
            return 0
        # Keep the process bound to the supervisor's failure domain.  A
        # forever-sleeping entry point would leave a stale READY marker when
        # the WS or a required worker dies after startup.
        await runtime.components.supervisor.wait()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        # 2026-09-15：只打类型会把 ReadinessError 这类"消息才是全部信息"的失败变成哑巴。
        print(f"start_stack: failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            await runtime.stop()
        else:
            await client.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(_args(argv)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
