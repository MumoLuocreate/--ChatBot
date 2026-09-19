"""Safe Windows logon startup registration for the Qichi runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

try:
    import winreg
except ImportError:  # pragma: no cover - only reached on non-Windows hosts
    winreg = None  # type: ignore[assignment]


_RUN_SUBKEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class AutostartError(RuntimeError):
    """The requested startup operation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class AutostartSpec:
    """One exact per-user Windows logon registration."""

    project_root: Path
    task_name: str = "QichiCompanion"

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        if not root.is_dir():
            raise AutostartError("project root does not exist")
        if not self.task_name or any(char in self.task_name for char in '\"\r\n'):
            raise AutostartError("task name is invalid")
        object.__setattr__(self, "project_root", root)

    @property
    def interpreter(self) -> Path:
        return self.project_root / ".venv" / "Scripts" / "python.exe"

    @property
    def start_script(self) -> Path:
        return self.project_root / "scripts" / "start_stack.py"

    @property
    def launcher_script(self) -> Path:
        """Single production entry point that starts NapCat before Qichi."""
        return self.project_root / "scripts" / "start_production.ps1"

    @property
    def config(self) -> Path:
        return self.project_root / "config.example.yaml"

    @property
    def evidence(self) -> Path:
        """按配置里的主模型取能力证据（2026-09-12 起支持 pro）。

        这里只做文件存在性检查，不能因为读模型名而要求 API key：所以直接读配置文本里
        的 `model:` 行，读不到就按 flash 兜底（真跑起来时 runtime 仍会自己 fail closed）。
        """

        model = "deepseek-v4-flash"
        try:
            text = self.config.read_text(encoding="utf-8")
        except OSError:
            text = ""
        match = re.search(r"^\s*model:\s*(\S+)\s*$", text, re.MULTILINE)
        if match is not None:
            model = match.group(1)
        return self.project_root / "runtime" / f"{model}-capability.json"

    @property
    def ready_marker(self) -> Path:
        return self.project_root / "runtime" / "qichi-ready.json"

    @property
    def lock(self) -> Path:
        return self.project_root / "runtime" / "qichi.lock"

    @staticmethod
    def _quote(path: Path) -> str:
        # schtasks receives /TR as one argument; quote each absolute path for
        # spaces without invoking a shell.
        return f'"{path}"'

    def start_command(self) -> str:
        return " ".join(
            (
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-WindowStyle",
                "Hidden",
                "-File",
                self._quote(self.launcher_script),
            )
        )

    def _ensure_windows(self) -> None:
        if os.name != "nt":
            raise AutostartError("Windows startup registration is only available on Windows")

    def _ensure_files(self) -> None:
        required = (self.interpreter, self.start_script, self.launcher_script, self.config, self.evidence)
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise AutostartError("startup files are missing")

    @staticmethod
    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["schtasks.exe", *args],
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
        except OSError as error:
            raise AutostartError("schtasks.exe could not be started") from error
        return result

    @staticmethod
    def _registry() -> object:
        if winreg is None:
            raise AutostartError("Windows registry is unavailable")
        return winreg

    def _registry_value(self) -> str | None:
        registry = self._registry()
        try:
            with registry.OpenKey(registry.HKEY_CURRENT_USER, _RUN_SUBKEY) as key:
                value, _value_type = registry.QueryValueEx(key, self.task_name)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise AutostartError("current-user startup entry could not be read") from error
        return value if isinstance(value, str) else None

    def _registry_install(self) -> None:
        registry = self._registry()
        try:
            with registry.CreateKeyEx(
                registry.HKEY_CURRENT_USER,
                _RUN_SUBKEY,
                0,
                registry.KEY_SET_VALUE,
            ) as key:
                registry.SetValueEx(
                    key,
                    self.task_name,
                    0,
                    registry.REG_SZ,
                    self.start_command(),
                )
        except OSError as error:
            raise AutostartError("current-user startup entry could not be written") from error

    def _registry_remove(self, *, only_if_ours: bool = False) -> bool:
        registry = self._registry()
        current = self._registry_value()
        if current is None or (only_if_ours and current != self.start_command()):
            return False
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                _RUN_SUBKEY,
                0,
                registry.KEY_SET_VALUE,
            ) as key:
                registry.DeleteValue(key, self.task_name)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise AutostartError("current-user startup entry could not be removed") from error
        return True

    def install(self) -> bool:
        """Create or replace the current-user logon task without secrets."""
        self._ensure_windows()
        self._ensure_files()
        try:
            result = self._run(
                [
                    "/Create",
                    "/TN",
                    self.task_name,
                    "/SC",
                    "ONLOGON",
                    "/RL",
                    "LIMITED",
                    "/F",
                    "/TR",
                    self.start_command(),
                ]
            )
        except AutostartError:
            result = None
        if result is not None and result.returncode == 0:
            # A previous permission fallback may have left a duplicate Run
            # entry.  Remove it only when it is exactly ours.
            self._registry_remove(only_if_ours=True)
            return True

        # Some managed Windows installations deny schtasks even for
        # /RL LIMITED.  HKCU Run is still a per-user logon mechanism and does
        # not require elevation, so use it as the explicit fallback.
        self._registry_install()
        return True

    def status(self) -> bool:
        """Return whether the exact task name is registered."""
        self._ensure_windows()
        try:
            result = subprocess.run(
                ["schtasks.exe", "/Query", "/TN", self.task_name, "/FO", "LIST", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
        except OSError as error:
            result = None
        if result is not None and result.returncode == 0:
            return True
        return self._registry_value() == self.start_command()

    def remove(self) -> bool:
        """Remove the exact task, returning false when it was absent."""
        self._ensure_windows()
        try:
            result = subprocess.run(
                ["schtasks.exe", "/Delete", "/TN", self.task_name, "/F"],
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
        except OSError as error:
            result = None
        removed_task = result is not None and result.returncode == 0
        # Remove only the exact command written by this project; never delete
        # a user's unrelated value that happens to share the task name.
        removed_run = self._registry_remove(only_if_ours=True)
        return removed_task or removed_run
