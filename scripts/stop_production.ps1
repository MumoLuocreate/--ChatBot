<#
.SYNOPSIS
    Stop only the Qichi processes recorded by the production runtime.

.DESCRIPTION
    The script is intentionally evidence-driven.  It never kills by image
    name, never touches an unrecorded QQ process, and leaves the database and
    READY/lock files for the next startup to archive or validate.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    # The script lives under <project>\scripts; direct invocation should
    # resolve the project root rather than treating the scripts directory as
    # the database/runtime root.  Batch entry points still pass an explicit
    # root and remain unchanged.
    $ProjectRoot = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".."
}
$root = (Resolve-Path $ProjectRoot -ErrorAction Stop).Path
$runtime = Join-Path $root "runtime"
$stopped = @()
$skipped = @()

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        $raw = Get-Content -Raw -LiteralPath $Path
        # PowerShell 7 can preserve ISO timestamps as strings.  Windows
        # PowerShell 5.1 has no -DateKind; the fallback below keeps DateTime
        # objects and Stop-RecordedProcess normalizes them explicitly.
        try {
            return ($raw | ConvertFrom-Json -DateKind String -ErrorAction Stop)
        } catch {
            return ($raw | ConvertFrom-Json -ErrorAction Stop)
        }
    } catch {
        $script:skipped += "invalid:$Path"
        return $null
    }
}

function Convert-EvidenceTimeUtc([object]$Value) {
    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [DateTimeOffset]) {
        return $Value.ToUniversalTime()
    }
    if ($Value -is [DateTime]) {
        $date = [DateTime]$Value
        if ($date.Kind -eq [DateTimeKind]::Utc -or $date.Kind -eq [DateTimeKind]::Local) {
            return ([DateTimeOffset]$date).ToUniversalTime()
        }
        # *_utc fields are explicitly UTC when a legacy JSON parser produces
        # an Unspecified DateTime object.
        return [DateTimeOffset]::new($date, [TimeSpan]::Zero)
    }
    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    return [DateTimeOffset]::Parse(
        $text,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::AssumeUniversal
    ).ToUniversalTime()
}

function Stop-RecordedProcess([int]$ProcessId, [string]$ExpectedPath, [object]$ExpectedStart) {
    if ($ProcessId -le 0) {
        return
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }
    $expectedName = [IO.Path]::GetFileNameWithoutExtension($ExpectedPath)
    if (-not [string]::IsNullOrWhiteSpace($expectedName) -and
        -not ([string]$process.ProcessName).Equals($expectedName, [System.StringComparison]::OrdinalIgnoreCase)) {
        $script:skipped += "name-mismatch:$ProcessId"
        return
    }
    try {
        $actualPath = [string]$process.Path
    } catch {
        $actualPath = ""
    }
    if (-not [string]::IsNullOrWhiteSpace($actualPath) -and
        -not [string]::IsNullOrWhiteSpace($ExpectedPath) -and
        -not $actualPath.Equals($ExpectedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        $script:skipped += "path-mismatch:$ProcessId"
        return
    }
    if ($null -ne $ExpectedStart -and -not [string]::IsNullOrWhiteSpace([string]$ExpectedStart)) {
        try {
            $expected = Convert-EvidenceTimeUtc $ExpectedStart
            $actual = [DateTimeOffset]$process.StartTime.ToUniversalTime()
            if ([Math]::Abs(($actual - $expected).TotalSeconds) -gt 3) {
                $script:skipped += "start-mismatch:$ProcessId"
                return
            }
        } catch {
            $script:skipped += "start-unavailable:$ProcessId"
            return
        }
    }
    if ([string]::IsNullOrWhiteSpace($actualPath)) {
        # The exact recorded process name and UTC start time are the remaining
        # evidence when Windows hides the executable path for QQ children.
        Write-Output ("qichi stop: path unavailable for recorded {0} PID {1}; name/time evidence matched" -f $expectedName, $ProcessId)
    }
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        $script:stopped += $ProcessId
    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
        # A recorded child may exit between the evidence check and the stop
        # call.  Treat that race as idempotent success; never broaden the
        # target set to compensate for a missing PID.
        if ($null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            $script:stopped += $ProcessId
        } else {
            $script:skipped += "stop-failed:$ProcessId"
        }
    } catch {
        $script:skipped += "stop-failed:$ProcessId"
    }
}

function Stop-RecordedCommandProcess([int]$ProcessId, [string]$CommandNeedle) {
    if ($ProcessId -le 0) {
        return
    }
    try {
        $record = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $ProcessId) -ErrorAction Stop
    } catch {
        $script:skipped += "inspect-failed:$ProcessId"
        return
    }
    if ($null -eq $record -or [string]$record.CommandLine -notlike ("*{0}*" -f $CommandNeedle)) {
        $script:skipped += "command-mismatch:$ProcessId"
        return
    }
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        $script:stopped += $ProcessId
    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
        # A child can exit between the CIM read and Stop-Process.  Treat that
        # as idempotent success only when the process no longer exists.
        if ($null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            $script:stopped += $ProcessId
        } else {
            $script:skipped += "stop-failed:$ProcessId"
        }
    } catch {
        $script:skipped += "stop-failed:$ProcessId"
    }
}

$ready = Read-JsonFile (Join-Path $runtime "qichi-ready.json")
if ($null -ne $ready) {
    $stackNeedle = (Join-Path $root "scripts\start_stack.py")
    foreach ($runtimePid in @([int]$ready.pid, [int]$ready.parent_pid)) {
        Stop-RecordedCommandProcess $runtimePid $stackNeedle
    }
}

$productionNeedle = (Join-Path $root "scripts\start_production.ps1")
$dashboardNeedle = (Join-Path $root "scripts\dashboard_server.py")
try {
    $projectProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $command = [string]$_.CommandLine
        ($command -like ("*{0}*" -f $productionNeedle)) -or
        ($command -like ("*{0}*" -f $dashboardNeedle))
    })
    foreach ($record in $projectProcesses) {
        $command = [string]$record.CommandLine
        if ($command -like ("*{0}*" -f $productionNeedle)) {
            Stop-RecordedCommandProcess ([int]$record.ProcessId) $productionNeedle
        } elseif ($command -like ("*{0}*" -f $dashboardNeedle)) {
            Stop-RecordedCommandProcess ([int]$record.ProcessId) $dashboardNeedle
        }
    }
} catch {
    $script:skipped += "project-process-scan-failed"
}

$napcat = Read-JsonFile (Join-Path $runtime "napcat-processes.json")
# ConvertFrom-Json returns a PSCustomObject for one recorded process and an
# array only when there are multiple processes.  Normalize both shapes before
# iterating; otherwise a single QQ child is silently skipped during stop.
if ($null -ne $napcat -and $null -ne $napcat.processes) {
    $qqPath = [string]$napcat.executable
    foreach ($record in @($napcat.processes)) {
        Stop-RecordedProcess ([int]$record.pid) $qqPath $record.start_time
    }
}
if ($null -ne $napcat -and $null -ne $napcat.launcher_pid) {
    $launcherPath = [string]$napcat.launcher_executable
    if ([string]::IsNullOrWhiteSpace($launcherPath) -and -not [string]::IsNullOrWhiteSpace([string]$napcat.executable)) {
        $launcherPath = Join-Path (Split-Path -Parent ([string]$napcat.executable)) "NapCatWinBootMain.exe"
    }
    Stop-RecordedProcess ([int]$napcat.launcher_pid) $launcherPath $napcat.launcher_start_time
}

if ($stopped.Count -eq 0) {
    Write-Output "qichi stop: no recorded live process stopped"
} else {
    Write-Output ("qichi stop: stopped {0} recorded process(es)" -f $stopped.Count)
}
if ($skipped.Count -gt 0) {
    Write-Output ("qichi stop: skipped {0} target(s) after evidence mismatch" -f $skipped.Count)
}
