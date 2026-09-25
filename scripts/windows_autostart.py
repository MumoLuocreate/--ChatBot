"""Install, inspect, or remove Qichi's per-user Windows logon task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.autostart import AutostartError, AutostartSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "status", "remove"))
    args = parser.parse_args(argv)
    spec = AutostartSpec(_ROOT)
    try:
        if args.action == "install":
            spec.install()
            print(f"autostart: installed ({spec.task_name})")
        elif args.action == "status":
            print(f"autostart: {'installed' if spec.status() else 'not installed'} ({spec.task_name})")
        else:
            print(f"autostart: {'removed' if spec.remove() else 'not present'} ({spec.task_name})")
    except AutostartError as error:
        print(f"autostart: failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
