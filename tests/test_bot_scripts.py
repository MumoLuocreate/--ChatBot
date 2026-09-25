from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_start_bot_uses_the_single_production_entry_and_explicit_qq_path():
    source = (ROOT / "start_bot.bat").read_text(encoding="ascii")

    assert "start_production.ps1" in source
    assert "start_stack.py" not in source
    assert "QICHI_NAPCAT_ROOT=E:\\NapCatQQ" in source
    assert "QICHI_QQ_EXE=E:\\QQ\\QQ.exe" in source
    assert "-WindowStyle Hidden" in source
    assert "current user session" in source
    assert "-Verb RunAs" not in source


def test_stop_bot_uses_evidence_driven_stop_script():
    batch = (ROOT / "stop_bot.bat").read_text(encoding="ascii")
    script = (ROOT / "scripts" / "stop_production.ps1").read_text(encoding="ascii")

    assert "stop_production.ps1" in batch
    assert "taskkill" not in batch.lower()
    assert "taskkill" not in script.lower()
    assert "napcat-processes.json" in script
    assert "launcher_pid" in script
    assert "NapCatWinBootMain.exe" in script
    assert "Path-mismatch" in script or "path-mismatch" in script
    assert "Stop-Process -Id $ProcessId" in script
    assert "[int]$Pid" not in script
    assert "function Stop-RecordedCommandProcess" in script
    assert "scripts\\start_stack.py" in script
    assert "scripts\\start_production.ps1" in script
    assert "scripts\\dashboard_server.py" in script
    assert "if ($command -like (\"*{0}*\" -f $productionNeedle))" in script
    assert "ProcessCommandException" in script
    assert "Get-Process -Name" not in script
    assert "-Verb RunAs" not in (ROOT / "stop_bot.bat").read_text(encoding="ascii")


def test_stop_bot_removes_trailing_backslash_before_passing_project_root_to_powershell():
    batch = (ROOT / "stop_bot.bat").read_text(encoding="ascii")

    assert "PROJECT_ROOT_NO_SLASH=%PROJECT_ROOT:~0,-1%" in batch
    assert '-ProjectRoot "%PROJECT_ROOT_NO_SLASH%"' in batch


def test_stop_script_parses_utc_evidence_and_allows_hidden_process_paths_only_with_name_and_time_match():
    script = (ROOT / "scripts" / "stop_production.ps1").read_text(encoding="ascii")

    assert "DateTimeOffset]::Parse" in script
    assert "path unavailable" in script.lower()
    assert "ProcessName" in script
    assert "start-mismatch" in script


def test_stop_script_treats_recorded_process_exit_race_as_idempotent():
    script = (ROOT / "scripts" / "stop_production.ps1").read_text(encoding="ascii")
    function = script[script.index("function Stop-RecordedProcess"):script.index("function Stop-RecordedCommandProcess")]

    assert "ProcessCommandException" in function
    assert 'Get-Process -Id $ProcessId -ErrorAction SilentlyContinue' in function
    assert 'stop-failed:$ProcessId' in function


def test_stop_script_resolves_its_own_root_when_called_without_project_root():
    script = (ROOT / "scripts" / "stop_production.ps1").read_text(encoding="ascii")

    assert '[string]$ProjectRoot = ""' in script
    assert "$MyInvocation.MyCommand.Path" in script
    assert 'Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".."' in script
