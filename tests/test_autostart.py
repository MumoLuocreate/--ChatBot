from __future__ import annotations

from pathlib import Path

import pytest

import qichi.autostart as autostart


def make_root(tmp_path: Path) -> Path:
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("", encoding="ascii")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "start_stack.py").write_text("", encoding="ascii")
    (tmp_path / "scripts" / "start_production.ps1").write_text("", encoding="ascii")
    (tmp_path / "config.example.yaml").write_text("", encoding="ascii")
    (tmp_path / "runtime").mkdir(parents=True)
    (tmp_path / "runtime" / "deepseek-v4-flash-capability.json").write_text("{}", encoding="ascii")
    return tmp_path


def test_spec_uses_repo_paths_and_never_embeds_secrets(tmp_path):
    root = make_root(tmp_path)
    spec = autostart.AutostartSpec(root)
    command = spec.start_command()

    assert str(root) in command
    assert "start_production.ps1" in command
    assert "start_stack.py" not in command
    assert "API_KEY" not in command
    assert "token" not in command.casefold()


def test_autostart_command_does_not_follow_registered_default_qq(tmp_path):
    root = make_root(tmp_path)
    command = autostart.AutostartSpec(root).start_command()

    assert "powershell.exe" in command.casefold()
    assert "E:\\QQ" not in command
    assert "NapCat" not in command
    assert "QICHI_NAPCAT_ROOT" not in command


def test_install_calls_schtasks_with_absolute_project_command(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    spec = autostart.AutostartSpec(root)
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append(args)
        return Completed()

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    assert spec.install() is True

    assert calls and calls[0][0].casefold() == "schtasks.exe"
    assert "/Create" in calls[0]
    assert "/SC" in calls[0] and calls[0][calls[0].index("/SC") + 1] == "ONLOGON"
    assert "/RU" not in calls[0]
    assert spec.task_name in calls[0]
    assert spec.start_command() in calls[0]


def test_status_and_remove_are_idempotent_for_missing_task(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    spec = autostart.AutostartSpec(root)

    class Completed:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "/Query" in args or "/Delete" in args:
            return Completed(1)
        return Completed(0)

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    assert spec.status() is False
    assert spec.remove() is False
    assert any("/Delete" in call for call in calls)


def test_non_windows_is_fail_closed(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    spec = autostart.AutostartSpec(root)
    monkeypatch.setattr(autostart.os, "name", "posix")

    with pytest.raises(autostart.AutostartError, match="Windows"):
        spec.install()


class _FakeRunKey:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeWinreg:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self):
        self.values: dict[str, str] = {}

    def OpenKey(self, _root, _subkey, *_args):
        if not self.values:
            raise FileNotFoundError
        return _FakeRunKey(self.values)

    def CreateKeyEx(self, _root, _subkey, *_args):
        return _FakeRunKey(self.values)

    def QueryValueEx(self, key, name):
        if name not in key.values:
            raise FileNotFoundError
        return key.values[name], self.REG_SZ

    def SetValueEx(self, key, name, _reserved, _value_type, value):
        key.values[name] = value

    def DeleteValue(self, key, name):
        if name not in key.values:
            raise FileNotFoundError
        del key.values[name]


def test_install_falls_back_to_current_user_run_when_scheduler_is_denied(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    spec = autostart.AutostartSpec(root)
    registry = _FakeWinreg()
    calls: list[list[str]] = []

    class Completed:
        returncode = 1
        stdout = "ERROR: Access is denied."
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append(args)
        return Completed()

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    monkeypatch.setattr(autostart, "winreg", registry)

    assert spec.install() is True
    assert calls[0][0].casefold() == "schtasks.exe"
    assert registry.values[spec.task_name] == spec.start_command()
    assert spec.status() is True
    assert spec.remove() is True
    assert spec.status() is False
