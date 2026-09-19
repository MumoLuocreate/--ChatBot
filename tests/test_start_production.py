from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "start_production.ps1"
_STOP_SCRIPT = SCRIPT.parent / "stop_production.ps1"

_PARSE_TEMPLATE = (
    "$e = $null; "
    "$null = [System.Management.Automation.Language.Parser]::ParseFile(PATH, [ref]$null, [ref]$e); "
    "if ($e.Count -gt 0) { $e | ForEach-Object { $_.Message }; exit 1 } else { exit 0 }"
)


def test_launcher_scripts_parse_as_windows_powershell():
    """A syntax error here means the logon entry point cannot start at all."""
    if sys.platform != "win32" or shutil.which("powershell") is None:
        pytest.skip("Windows PowerShell is required to parse the launcher scripts")
    for path in (SCRIPT, _STOP_SCRIPT):
        literal = "'" + str(path).replace("'", "''") + "'"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PARSE_TEMPLATE.replace("PATH", literal)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{path.name}: {result.stdout}{result.stderr}"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_start_entry_waits_for_delayed_login_and_has_a_bounded_window():
    source = _script()

    assert "[int]$LoginWaitSeconds = 900" in source
    assert "[int]$PollSeconds = 2" in source
    assert "NapCat login/endpoint readiness timed out" in source
    assert "Test-ProductionLogin" in source
    assert "Start-Sleep -Seconds ([Math]::Max(1, $PollSeconds))" in source


def test_start_entry_records_safe_phase_evidence_without_response_or_token():
    source = _script()
    log_function_start = source.index("function Write-StartLog")
    log_function_end = source.index("function Stop-Production")
    log_function = source[log_function_start:log_function_end]

    assert "runtime\\start-production.log" in source
    assert "Write-StartLog" in source
    assert "$accessToken" not in log_function
    assert "$response" not in log_function
    assert "Invoke-RestMethod" in source


def test_start_entry_keeps_single_handoff_to_stack():
    source = _script()

    assert source.count("$python $stack") == 1
    assert "start_stack: READY" not in source


def test_start_entry_starts_only_project_dashboard_without_blocking_core():
    source = _script()
    assert "function Start-LocalDashboard" in source
    assert "scripts\\dashboard_server.py" in source
    assert '"--port", "8765"' in source
    assert "--port\\s+8765" in source
    assert "dashboard already running for this project" in source
    assert "dashboard port occupied by non-project process" in source
    assert "dashboard start failed; continuing core handoff" in source
    assert "-WindowStyle Hidden" in source
    assert "-RedirectStandardOutput" in source and "-RedirectStandardError" in source


def test_start_entry_does_not_use_global_python_kill_or_log_response_content():
    source = _script()
    assert "taskkill" not in source.lower()
    assert "Stop-Process -Id $ProcessId" in source
    assert "dashboard.stdout.log" in source and "dashboard.stderr.log" in source


def test_start_entry_has_a_precise_process_probe_fallback_when_cim_is_denied():
    source = _script()

    assert 'Get-Process -Name "QQ"' in source
    assert 'qq\\QQ.exe' in source
    assert "OrdinalIgnoreCase" in source
    assert "isolated profile" in source


def test_start_entry_does_not_treat_default_qq_profile_as_napcat_instance():
    source = _script()

    probe = source[source.index("function Test-RootNapCatProcess"):source.index("function Test-ProductionLogin")]
    assert "profileNeedle" in probe
    assert "commandLine.IndexOf($profileNeedle" in probe
    assert "path.Equals($qqPath" in probe
    assert "path-only" in probe
    assert "NapCatWinBootMain.exe" in probe
    assert "ParentProcessId" in probe


def test_start_entry_filters_target_process_evidence_by_isolated_profile():
    source = _script()

    probe = source[source.index("function Get-TargetQQProcesses"):source.index("function Write-NapCatProcessEvidence")]
    assert "$ProfilePath" in probe
    assert "profileNeedle" in probe
    assert "CommandLine" in probe
    assert "default QQ profile" in probe
    assert "ParentProcessId" in probe
    assert "ownedIds" in probe


def test_start_entry_records_the_boot_launcher_so_failed_hooks_are_reapable():
    source = _script()

    assert "NapCatWinBootMain.exe" in source
    assert "-PassThru" in source
    assert "launcher_pid" in source
    assert "launcher_start_time" in source


def test_start_entry_does_not_block_non_elevated_logon_startup():
    source = _script()

    assert "Test-ElevatedSession" not in source
    assert "elevated Windows session is required for NapCat hook injection" not in source


def test_start_entry_builds_a_patch_package_from_the_installed_qq_version():
    source = _script()

    assert "function Write-NapCatPatchPackage" in source
    assert "versions\\config.json" in source
    assert "resources\\app\\package.json" in source
    assert 'patch["main"] = "./loadNapCat.js"' in source
    assert "NAPCAT_PATCH_PACKAGE" in source
    assert "QQ package metadata does not match versions/config.json" in source
    assert "UTF8Encoding]::new($false)" in source


def test_start_entry_records_actual_launcher_and_reaps_stuck_launcher():
    source = _script()

    assert "function Test-RecordedLauncherAlive" in source
    assert "function Stop-RecordedLauncher" in source
    assert "launcher_executable = $LauncherPath" in source
    assert "NapCat launcher stayed alive but target QQ did not start" in source
    assert "[int]$Pid" not in source


def test_patch_builder_does_not_mix_log_output_into_patch_path():
    source = _script()

    assert 'Write-StartLog ("NapCat patch package matched QQ {0}" -f $version) | Out-Null' in source


def test_start_entry_binds_napcat_quick_login_to_the_configured_bot_account():
    source = _script()

    assert "function Get-NapCatQuickLoginAccount" in source
    assert 'QICHI_BOT_QQ' in source
    assert 'onebot11_*.json' in source
    assert 'NAPCAT_QUICK_ACCOUNT = $quickLoginAccount' in source
    assert '"-q", $quickLoginAccount' in source
    assert 'NapCat quick-login account is ambiguous' in source


def test_start_entry_does_not_guess_quick_login_from_a_stale_ready_marker():
    source = _script()
    function = source[source.index("function Get-NapCatQuickLoginAccount"):source.index("function Test-RootNapCatProcess")]

    assert 'qichi-ready.json' not in function
    assert 'QICHI_BOT_QQ' in function
    assert 'onebot11_*.json' in function


def test_start_entry_can_use_bounded_recorded_process_evidence_when_windows_hides_paths():
    source = _script()

    assert "function Test-RecordedNapCatEvidence" in source
    assert "napcat-processes.json" in source
    assert "DateTimeOffset]::Parse" in source
    assert "launcher_pid" in source
    assert "launcher_start_time" in source
    assert "start time mismatch" in source
    assert "recorded process evidence" in source


def test_start_entry_does_not_reuse_previous_generation_when_endpoints_are_down():
    source = _script()

    assert "if ($httpReady -and $wsReady)" in source
    assert "recorded launch generation" in source
    assert "old NapCat process remains while endpoints are down" in source


def test_start_entry_preserves_launcher_evidence_when_refreshing_an_existing_generation():
    source = _script()

    assert "$evidenceLauncherPid" in source
    assert "$evidenceLauncherStart" in source
    assert "evidenceLauncherPid -le 0" in source
    assert "launcher_pid = $evidenceLauncherPid" in source


def test_process_probe_logs_do_not_pollute_boolean_return_values():
    source = _script()

    assert 'using bounded recorded process evidence (protected path hidden)" | Out-Null' in source
    assert 'recovered current NapCat launcher metadata from bounded recorded process evidence" | Out-Null' in source


def test_start_entry_can_recover_missing_launcher_metadata_only_from_current_unique_process_set():
    source = _script()

    assert "function Recover-ExistingNapCatEvidence" in source
    assert "NapCatWinBootMain" in source
    assert "recorded QQ PID" in source
    assert "launcher metadata" in source
    assert "current endpoint listener" in source
    assert '$tmp = "$EvidencePath.tmp"' in source
    assert "Move-Item -LiteralPath $tmp -Destination $EvidencePath -Force" in source


def test_start_entry_records_observed_profile_instead_of_trusting_the_launch_argument():
    source = _script()

    assert "function Get-ObservedUserDataDir" in source
    assert "--user-data-dir=" in source
    assert "profile = $ProfilePath" in source
    assert "profile_observed = $observedProfile" in source
    assert "profile_evidence = $profileEvidence" in source
    assert "user_data_dir = (Get-ObservedUserDataDir ([int]$process.Id))" in source
    for state in ('"unverified"', '"matches"', '"differs"'):
        assert state in source
    assert "NapCat profile evidence {0}" in source


def test_recorded_profile_evidence_never_becomes_a_startup_gate():
    source = _script()
    writer = source[source.index("function Get-ObservedUserDataDir"):]
    verifier = source[source.index("function Test-RecordedNapCatEvidence"):source.index("function Recover-ExistingNapCatEvidence")]
    recovery = source[source.index("function Recover-ExistingNapCatEvidence"):source.index("function Get-TargetQQProcesses")]
    ownership_probe = source[source.index("function Test-RootNapCatProcess"):source.index("function Get-TargetQQProcesses")]

    # Recording an observation must not turn into a gate: today no evidence can
    # prove the requested profile is honoured, so gating would block a working
    # production stack instead of fixing anything.
    assert "profile_evidence" not in verifier
    assert "profile_evidence" not in recovery
    assert "never used as a gate" in writer
    assert "it is deliberately not evidence that the" in ownership_probe


def test_stop_entry_normalizes_single_recorded_napcat_process():
    stop_source = (SCRIPT.parents[1] / "scripts" / "stop_production.ps1").read_text(
        encoding="utf-8"
    )
    assert "$null -ne $napcat.processes" in stop_source
    assert "foreach ($record in @($napcat.processes))" in stop_source
    assert "$napcat.processes -is [System.Collections.IEnumerable]" not in stop_source
