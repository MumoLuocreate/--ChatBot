"""Store a supported provider API key through a local hidden-input GUI.

2026-09-14：终端里 getpass 在部分 Windows 控制台粘不进内容，所以这条通道不是
"DeepSeek 专用"，而是所有已登记 provider 共用的备用入口。值只在进程内存与当前
用户注册表之间移动：不写文件、不进日志、不进命令行参数。
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

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


# 只用于窗口文案；这里刻意不写环境变量名，避免把「密钥放在哪个变量」抄进界面。
_PROVIDER_LABELS = {
    "deepseek": "DeepSeek 官方（对话模型）",
    "dashscope": "阿里云百炼（角色的声音 / TTS）",
}


def _label_for(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider)


def _bind_paste(entry: tk.Entry, root: tk.Misc) -> None:
    """右键粘贴：Tk 默认不绑定，而这条通道存在的理由就是"粘不进去"。"""

    def paste(_event: object = None) -> str:
        try:
            value = root.clipboard_get()
        except tk.TclError:
            return "break"
        entry.delete(0, tk.END)
        entry.insert(0, value.strip())
        return "break"

    entry.bind("<Button-3>", paste)


def run(provider: str = "deepseek") -> int:
    try:
        require_windows()
        store = ApiCredentialStore(credential_for(provider))
    except ApiCredentialError as error:
        messagebox.showerror("角色 API 配置", str(error))
        return 1

    provider_name = store.credential.provider
    display = _label_for(provider_name)

    root = tk.Tk()
    root.title(f"角色 - {display} API 配置")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    frame = tk.Frame(root, padx=22, pady=18)
    frame.grid()
    tk.Label(frame, text=f"{display} API Key").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
    )
    tk.Label(frame, text="API Key").grid(row=1, column=0, sticky="w", pady=4)
    first = tk.Entry(frame, width=48, show="*", exportselection=False)
    first.grid(row=1, column=1, pady=4)
    tk.Label(frame, text="再次输入").grid(row=2, column=0, sticky="w", pady=4)
    second = tk.Entry(frame, width=48, show="*", exportselection=False)
    second.grid(row=2, column=1, pady=4)
    status = tk.Label(
        frame,
        text="Ctrl+V 或右键粘贴；输入不会显示，也不会写入聊天或日志",
        fg="#555555",
    )
    status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 12))
    _bind_paste(first, root)
    _bind_paste(second, root)

    def save() -> None:
        try:
            first_value = first.get()
            second_value = second.get()
            if first_value != second_value:
                raise ApiCredentialError("两次输入的 API Key 不一致")
            store.set(first_value)
        except ApiCredentialError as error:
            messagebox.showerror("保存失败", str(error), parent=root)
            return
        first.delete(0, tk.END)
        second.delete(0, tk.END)
        status.configure(text="已保存到当前 Windows 用户配置", fg="#176b3a")
        messagebox.showinfo("保存成功", f"{display} 的 API Key 已保存。", parent=root)
        root.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e")
    tk.Button(buttons, text="取消", command=root.destroy, width=10).pack(
        side="right", padx=(8, 0)
    )
    tk.Button(buttons, text="保存", command=save, width=10).pack(side="right")
    first.focus_set()
    root.bind("<Return>", lambda _event: save())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "deepseek"))
