from __future__ import annotations

from pathlib import Path

import pytest

import qichi.api_credentials as credentials


class FakeKey:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeWinreg:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self):
        self.values: dict[str, str] = {}

    def CreateKeyEx(self, _root, _subkey, *_args):
        return FakeKey(self.values)

    def OpenKey(self, _root, _subkey, *_args):
        return FakeKey(self.values)

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


def test_deepseek_credential_uses_current_user_registry_without_revealing_value(monkeypatch):
    fake = FakeWinreg()
    monkeypatch.setattr(credentials, "winreg", fake)
    store = credentials.ApiCredentialStore(credentials.credential_for("DeepSeek"))

    store.set("  secret-value  ")

    assert store.exists() is True
    assert fake.values == {"DEEPSEEK_API_KEY": "secret-value"}
    assert store.remove() is True
    assert store.exists() is False


@pytest.mark.parametrize("value", ["", "  ", "bad\nkey", "bad\rkey", "bad\x00key"])
def test_secret_validation_rejects_empty_or_control_characters(monkeypatch, value):
    fake = FakeWinreg()
    monkeypatch.setattr(credentials, "winreg", fake)
    store = credentials.ApiCredentialStore(credentials.credential_for("deepseek"))

    with pytest.raises(credentials.ApiCredentialError):
        store.set(value)


def test_remove_is_idempotent(monkeypatch):
    fake = FakeWinreg()
    monkeypatch.setattr(credentials, "winreg", fake)
    store = credentials.ApiCredentialStore(credentials.credential_for("deepseek"))

    assert store.remove() is False
    store.set("secret")
    assert store.remove() is True
    assert store.remove() is False


def test_non_windows_is_fail_closed(monkeypatch):
    monkeypatch.setattr(credentials.os, "name", "posix")
    with pytest.raises(credentials.ApiCredentialError, match="Windows"):
        credentials.require_windows()


def test_unknown_provider_is_rejected():
    with pytest.raises(credentials.ApiCredentialError, match="unsupported"):
        credentials.credential_for("siliconflow")


def test_dashscope_credential_shares_the_same_hidden_input_channel(monkeypatch):
    """2026-09-14：角色的声音走阿里云百炼

    声音设计需要另一家供应商的密钥，但必须复用同一条通道：值由 getpass 隐藏输入，
    写当前用户注册表，既不留 shell 历史也不进仓库。
    """

    fake = FakeWinreg()
    monkeypatch.setattr(credentials, "winreg", fake)
    store = credentials.ApiCredentialStore(credentials.credential_for("DashScope"))

    store.set("  tts-secret  ")

    assert store.credential.env_name == "DASHSCOPE_API_KEY"
    assert fake.values == {"DASHSCOPE_API_KEY": "tts-secret"}
    assert store.exists() is True
    assert store.remove() is True
    assert store.exists() is False


def test_gui_entrypoint_is_present_and_uses_hidden_entries():
    source = (Path(__file__).parents[1] / "scripts" / "provide_api_gui.py").read_text(
        encoding="utf-8"
    )
    assert 'show="*"' in source
    assert "store.set(first_value)" in source
    assert "DEEPSEEK_API_KEY" not in source


def test_gui_is_shared_by_every_registered_provider():
    """2026-09-14：终端里 getpass 在部分控制台粘不进内容。

    用户报告「我在终端里无法粘贴 key」。GUI 因此不再只服务 deepseek，
    而是所有已登记 provider 共用的备用入口；界面文案里仍然不出现环境变量名。
    """

    scripts = Path(__file__).parents[1] / "scripts"
    gui = (scripts / "provide_api_gui.py").read_text(encoding="utf-8")
    cli = (scripts / "provide_api.py").read_text(encoding="utf-8")

    assert "def run(provider" in gui
    assert "run(args.provider)" in cli
    assert "deepseek only" not in cli
    assert 'show="*"' in gui
    assert "store.set(first_value)" in gui
    assert "<Button-3>" in gui
    for env_name in (item.env_name for item in credentials.SUPPORTED_PROVIDERS.values()):
        assert env_name not in gui
        assert env_name not in cli
