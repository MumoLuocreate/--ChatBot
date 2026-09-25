"""Provision a supported provider API key without exposing it in the shell."""

from __future__ import annotations

import argparse
import getpass
import sys

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.api_credentials import (  # noqa: E402
    SUPPORTED_PROVIDERS,
    ApiCredentialError,
    ApiCredentialStore,
    credential_for,
    require_windows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("set", "gui", "status", "remove"))
    parser.add_argument(
        "--provider",
        default="deepseek",
        help="provider name (" + ", ".join(sorted(SUPPORTED_PROVIDERS)) + ")",
    )
    args = parser.parse_args(argv)

    try:
        require_windows()
        store = ApiCredentialStore(credential_for(args.provider))
        if args.action == "gui":
            # 2026-09-14：终端里 getpass 在部分控制台粘不进内容，所以 GUI 不再
            # 只服务 deepseek —— 它是所有已登记 provider 共用的备用入口。
            from provide_api_gui import run

            return run(args.provider)
        if args.action == "set":
            first = getpass.getpass(f"{store.credential.provider} API key (input hidden): ")
            second = getpass.getpass("Repeat API key (input hidden): ")
            if first != second:
                raise ApiCredentialError("API keys do not match")
            store.set(first)
            print(f"api: stored ({store.credential.env_name}, current user)")
        elif args.action == "status":
            print(f"api: {'configured' if store.exists() else 'not configured'} ({store.credential.provider})")
        else:
            print(
                f"api: {'removed' if store.remove() else 'not configured'} "
                f"({store.credential.provider})"
            )
    except ApiCredentialError as error:
        print(f"api: failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
