"""Secure per-user API credential provisioning for supported providers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

try:
    import winreg
except ImportError:  # pragma: no cover - only reached on non-Windows hosts
    winreg = None  # type: ignore[assignment]


_ENVIRONMENT_SUBKEY = r"Environment"


class ApiCredentialError(RuntimeError):
    """The credential could not be stored or queried safely."""


@dataclass(frozen=True, slots=True)
class ProviderCredential:
    """The non-secret storage contract for one provider."""

    provider: str
    env_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be non-empty text")
        if not isinstance(self.env_name, str) or not self.env_name.strip():
            raise ValueError("environment variable name must be non-empty text")
        if not self.env_name.replace("_", "").isalnum() or self.env_name.upper() != self.env_name:
            raise ValueError("environment variable name must be uppercase ASCII")


# 2026-09-14：TTS（阿里云百炼「声音设计」）需要另一家供应商的密钥通道。
# 这里只登记「供应商名 -> 环境变量名」；值永远由用户经 getpass 输入，进注册表，
# 不进仓库、不进日志、不进文档。仓库其余部分只按环境变量名引用。
SUPPORTED_PROVIDERS: dict[str, ProviderCredential] = {
    "deepseek": ProviderCredential("deepseek", "DEEPSEEK_API_KEY"),
    "dashscope": ProviderCredential("dashscope", "DASHSCOPE_API_KEY"),
}


class ApiCredentialStore:
    """Store one provider key under the current user's environment only."""

    def __init__(self, credential: ProviderCredential):
        if not isinstance(credential, ProviderCredential):
            raise TypeError("credential must be a ProviderCredential")
        self.credential = credential

    @staticmethod
    def _registry() -> Any:
        if winreg is None:
            raise ApiCredentialError("Windows user environment is unavailable")
        return winreg

    @staticmethod
    def _validate_secret(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ApiCredentialError("API key must be non-empty")
        if any(char in value for char in "\r\n\x00"):
            raise ApiCredentialError("API key contains an invalid control character")
        return value.strip()

    def set(self, value: str) -> None:
        secret = self._validate_secret(value)
        registry = self._registry()
        try:
            with registry.CreateKeyEx(
                registry.HKEY_CURRENT_USER,
                _ENVIRONMENT_SUBKEY,
                0,
                registry.KEY_SET_VALUE,
            ) as key:
                registry.SetValueEx(
                    key,
                    self.credential.env_name,
                    0,
                    registry.REG_SZ,
                    secret,
                )
        except OSError as error:
            raise ApiCredentialError("API key could not be stored for the current user") from error

    def exists(self) -> bool:
        registry = self._registry()
        try:
            with registry.OpenKey(registry.HKEY_CURRENT_USER, _ENVIRONMENT_SUBKEY) as key:
                value, _value_type = registry.QueryValueEx(key, self.credential.env_name)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ApiCredentialError("API key could not be queried") from error
        return isinstance(value, str) and bool(value.strip())

    def remove(self) -> bool:
        registry = self._registry()
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                _ENVIRONMENT_SUBKEY,
                0,
                registry.KEY_SET_VALUE,
            ) as key:
                registry.DeleteValue(key, self.credential.env_name)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ApiCredentialError("API key could not be removed") from error
        return True


def credential_for(provider: str) -> ProviderCredential:
    if not isinstance(provider, str) or not provider.strip():
        raise ApiCredentialError("provider must be non-empty")
    try:
        return SUPPORTED_PROVIDERS[provider.strip().casefold()]
    except KeyError as error:
        raise ApiCredentialError("unsupported provider") from error


def require_windows() -> None:
    if os.name != "nt":
        raise ApiCredentialError("API credential provisioning is only available on Windows")
