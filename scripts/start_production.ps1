<#
.SYNOPSIS
    Start the isolated NapCat gateway and then the Qichi production stack.

.DESCRIPTION
    This is the only Windows logon entry point for production.  It deliberately
    does not discover QQ from the registry: the QQ executable is an explicit
    path, while the hook and the requested profile come from one explicit NapCat
    root.  The profile switch is only a launch request: observation has shown the
    running QQ may keep its default profile instead, so an isolated profile is
    never assumed from the argument -- the observed value is recorded in
    runtime\napcat-processes.json instead.  Once NapCat is healthy, the script
    hands control to start_stack.py, which owns the Forward WebSocket, SQLite
    recovery and READY marker.
#>

[CmdletBinding()]
param(
    [string]$NapCatRoot,
    [string]$QQExecutable,
    [int]$LoginWaitSeconds = 900,
    [int]$PollSeconds = 2,
    [int]$MaxNapCatRestarts = 1
)

$ErrorActionPreference = "Stop"
$script:StartLogPath = $null
$script:NapCatLauncherPid = 0
$script:NapCatLauncherStart = ""

function Write-StartLog([string]$Message) {
    # Keep startup evidence useful when the Run entry is hidden, without ever
    # serializing API responses, access tokens, or message content.
    $line = "{0} {1}" -f (Get-Date).ToString("o"), ("start_production: " + $Message)
    if (-not [string]::IsNullOrWhiteSpace([string]$script:StartLogPath)) {
        try {
            Add-Content -LiteralPath $script:StartLogPath -Value $line -Encoding UTF8 -ErrorAction Stop
        } catch {
            # Logging cannot make the production handoff fail.
        }
    }
    Write-Output $line
}

function Stop-Production([string]$Message) {
    Write-StartLog ("ERROR " + $Message)
    Write-Error ("start_production: " + $Message)
    exit 1
}

function Import-PersistedEnvironment {
    # A logon task normally receives these values from the user environment.
    # Reading HKCU here also covers an already-open shell after provide_api.py.
    if ($null -eq (Get-Command Get-ItemProperty -ErrorAction SilentlyContinue)) {
        return
    }
    try {
        $persisted = Get-ItemProperty -Path "HKCU:\Environment" -ErrorAction Stop
        foreach ($name in @(
            "QICHI_OWNER_QQ",
            "NAPCAT_WS_URL",
            "NAPCAT_HTTP_URL",
            "NAPCAT_ACCESS_TOKEN",
            "SILICONFLOW_API_KEY",
            "DEEPSEEK_API_KEY",
            # 2026-09-14 TTS: the Bailian key must be imported too, otherwise
            # voice.enabled=true fails config validation at startup.
            "DASHSCOPE_API_KEY"
        )) {
            $current = [Environment]::GetEnvironmentVariable($name, "Process")
            if (-not [string]::IsNullOrWhiteSpace([string]$current)) {
                continue
            }
            $value = $persisted.$name
            if ($value -is [string] -and -not [string]::IsNullOrWhiteSpace($value)) {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    } catch {
        # The regular process-level environment remains authoritative when the
        # registry is unavailable; start_stack.py will report missing config.
    }
}

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMilliseconds = 1000) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait($TimeoutMilliseconds)) {
            return $false
        }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-EndpointUri([string]$Value, [string]$DefaultValue, [string]$ExpectedScheme) {
    $candidate = if ([string]::IsNullOrWhiteSpace($Value)) { $DefaultValue } else { $Value.Trim() }
    try {
        $uri = [System.Uri]$candidate
    } catch {
        Stop-Production ("invalid NapCat endpoint")
    }
    if ($uri.Scheme -ne $ExpectedScheme -or -not ($uri.Host -in @("127.0.0.1", "localhost", "::1"))) {
        Stop-Production ("NapCat endpoint must be local $ExpectedScheme")
    }
    if ($uri.Port -le 0) {
        Stop-Production "NapCat endpoint has no valid port"
    }
    return $uri
}

function Write-NapCatPatchPackage([string]$QQExecutable, [string]$RuntimePath) {
    # NapCat's bundled qqnt.json can describe an older QQ shell.  Passing that
    # stale descriptor to the hook causes the QQ process to break before the
    # NapCat runtime has a chance to start.  Build the small patch descriptor
    # from the installed QQ package on every launch instead of mutating QQ or
    # trusting the registry version.
    try {
        $qqRoot = Split-Path -Parent ([IO.Path]::GetFullPath($QQExecutable))
        $versionConfigPath = Join-Path $qqRoot "versions\config.json"
        if (-not (Test-Path -LiteralPath $versionConfigPath -PathType Leaf)) {
            Stop-Production "QQ version metadata is missing"
        }
        $versionConfig = Get-Content -Raw -LiteralPath $versionConfigPath | ConvertFrom-Json -ErrorAction Stop
        $version = [string]$versionConfig.curVersion
        $build = [string]$versionConfig.buildId
        if ([string]::IsNullOrWhiteSpace($version) -or [string]::IsNullOrWhiteSpace($build)) {
            Stop-Production "QQ version metadata is incomplete"
        }
        $packagePath = Join-Path (Join-Path (Join-Path $qqRoot "versions") $version) "resources\app\package.json"
        if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
            Stop-Production "QQ package metadata is missing"
        }
        $package = Get-Content -Raw -LiteralPath $packagePath | ConvertFrom-Json -ErrorAction Stop
        if ([string]$package.version -ne $version -or [string]$package.buildVersion -ne $build) {
            Stop-Production "QQ package metadata does not match versions/config.json"
        }
        $patch = [ordered]@{}
        foreach ($property in $package.PSObject.Properties) {
            $patch[$property.Name] = $property.Value
        }
        $patch["version"] = $version
        $patch["buildVersion"] = $build
        $patch["main"] = "./loadNapCat.js"
        $patch["isPureShell"] = $true
        $patch["isByteCodeShell"] = $true
        $patch["platform"] = "win32"
        $patch["eleArch"] = "x64"
        $patchDir = Split-Path -Parent $RuntimePath
        New-Item -ItemType Directory -Path $patchDir -Force -ErrorAction Stop | Out-Null
        $tmp = "$RuntimePath.tmp"
        $json = $patch | ConvertTo-Json -Depth 8
        [IO.File]::WriteAllText($tmp, $json, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tmp -Destination $RuntimePath -Force -ErrorAction Stop
        Write-StartLog ("NapCat patch package matched QQ {0}" -f $version) | Out-Null
        return $RuntimePath
    } catch [System.Management.Automation.ExitException] {
        throw
    } catch {
        Stop-Production "could not build a matching NapCat patch package"
    }
}

function Get-NapCatQuickLoginAccount([string]$NapCatRoot) {
    # Quick-login material lives in the QQ account data directory
    # (Documents\Tencent Files\<uin>), not in the --user-data-dir profile:
    # 2026-09-10 read-only evidence shows the running QQ declares the default
    # profile ("C:\Users\<user>\AppData\Roaming\QQ") on its child command
    # lines and never opens the requested isolated profile.  NapCat only
    # consumes the account automatically when it is passed as
    # NAPCAT_QUICK_ACCOUNT (or configured in webui.json).  Resolve the account
    # from the project binding/config, never from an arbitrary QQ registry
    # install or a stale READY marker.
    $configured = [string]$env:QICHI_BOT_QQ
    if (-not [string]::IsNullOrWhiteSpace($configured)) {
        if ($configured -notmatch '^[0-9]+$') {
            Stop-Production "QICHI_BOT_QQ must be a decimal QQ number"
        }
        return $configured
    }
    $configDir = Join-Path $NapCatRoot "napcat\config"
    if (-not (Test-Path -LiteralPath $configDir -PathType Container)) {
        Stop-Production "NapCat config directory is missing"
    }
    $candidates = @(Get-ChildItem -LiteralPath $configDir -Filter "onebot11_*.json" -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.BaseName -match '^onebot11_([0-9]+)$') { $Matches[1] }
        } | Sort-Object -Unique)
    if ($candidates.Count -ne 1) {
        Stop-Production "NapCat quick-login account is ambiguous; set QICHI_BOT_QQ explicitly"
    }
    return [string]$candidates[0]
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

function Test-RecordedNapCatEvidence([string]$EvidencePath, [string]$ExpectedQQPath, [string]$ExpectedProfile, [string]$ExpectedLauncherPath) {
    # Protected QQ children can hide ExecutablePath and CommandLine even from
    # an elevated PowerShell.  A launch record is a bounded fallback only when
    # its schema, exact paths, PID names and UTC start times still match.
    if ([string]::IsNullOrWhiteSpace($EvidencePath) -or
        -not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) {
        return $false
    }
    try {
        $evidenceRaw = Get-Content -Raw -LiteralPath $EvidencePath
        try {
            $evidence = $evidenceRaw | ConvertFrom-Json -DateKind String -ErrorAction Stop
        } catch {
            $evidence = $evidenceRaw | ConvertFrom-Json -ErrorAction Stop
        }
        if ([string]$evidence.schema -ne "qichi-napcat-processes" -or
            -not ([string]$evidence.executable).Equals($ExpectedQQPath, [System.StringComparison]::OrdinalIgnoreCase) -or
            -not ([string]$evidence.profile).TrimEnd("\\").Equals($ExpectedProfile.TrimEnd("\\"), [System.StringComparison]::OrdinalIgnoreCase) -or
            -not ([string]$evidence.launcher_executable).Equals($ExpectedLauncherPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        $launcherPid = [int]$evidence.launcher_pid
        $launcher = Get-Process -Id $launcherPid -ErrorAction SilentlyContinue
        if ($null -eq $launcher -or $launcher.ProcessName -ne "NapCatWinBootMain") {
            return $false
        }
        $expectedLauncherStart = Convert-EvidenceTimeUtc $evidence.launcher_start_time
        if ($null -eq $expectedLauncherStart) {
            return $false
        }
        $actualLauncherStart = [DateTimeOffset]$launcher.StartTime.ToUniversalTime()
        if ([Math]::Abs(($actualLauncherStart - $expectedLauncherStart).TotalSeconds) -gt 3) {
            Write-StartLog "WARN recorded process evidence launcher start time mismatch" | Out-Null
            return $false
        }
        $validQQ = 0
        foreach ($record in @($evidence.processes)) {
            $processId = [int]$record.pid
            if ($processId -le 0) {
                continue
            }
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -eq $process -or $process.ProcessName -ne "QQ") {
                continue
            }
            try {
                $expectedStart = Convert-EvidenceTimeUtc $record.start_time
                if ($null -eq $expectedStart) {
                    continue
                }
                $actualStart = [DateTimeOffset]$process.StartTime.ToUniversalTime()
                if ([Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -gt 3) {
                    Write-StartLog ("WARN recorded process evidence start time mismatch for PID {0}" -f $processId) | Out-Null
                    continue
                }
            } catch {
                continue
            }
            try {
                $actualPath = [string]$process.Path
            } catch {
                $actualPath = ""
            }
            if (-not [string]::IsNullOrWhiteSpace($actualPath) -and
                -not $actualPath.Equals($ExpectedQQPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            $validQQ += 1
        }
        if ($validQQ -gt 0) {
            Write-StartLog "using bounded recorded process evidence (protected path hidden)" | Out-Null
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Recover-ExistingNapCatEvidence([string]$EvidencePath, [string]$ExpectedQQPath, [string]$ExpectedProfile, [string]$ExpectedLauncherPath) {
    # A previous refresh may have written launcher_pid=0 when the protected
    # launcher path was hidden. Recover only the current generation: exactly
    # one launcher, at least one recorded QQ PID with matching UTC start time,
    # and both current endpoint listener rows owned by one of those QQ PIDs.
    if (-not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) { return $false }
    try {
        $raw = Get-Content -Raw -LiteralPath $EvidencePath
        try { $evidence = $raw | ConvertFrom-Json -DateKind String -ErrorAction Stop }
        catch { $evidence = $raw | ConvertFrom-Json -ErrorAction Stop }
        if ([string]$evidence.schema -ne "qichi-napcat-processes" -or
            [int]$evidence.launcher_pid -gt 0 -or
            -not ([string]$evidence.executable).Equals($ExpectedQQPath, [System.StringComparison]::OrdinalIgnoreCase) -or
            -not ([string]$evidence.profile).TrimEnd("\").Equals($ExpectedProfile.TrimEnd("\"), [System.StringComparison]::OrdinalIgnoreCase) -or
            -not ([string]$evidence.launcher_executable).Equals($ExpectedLauncherPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        $launchers = @(Get-Process -Name "NapCatWinBootMain" -ErrorAction SilentlyContinue)
        if ($launchers.Count -ne 1) { return $false }
        $validQQ = @()
        foreach ($record in @($evidence.processes)) {
            $processId = [int]$record.pid
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -eq $process -or $process.ProcessName -ne "QQ") { continue }
            try {
                $expectedStart = Convert-EvidenceTimeUtc $record.start_time
                $actualStart = [DateTimeOffset]$process.StartTime.ToUniversalTime()
                if ($null -eq $expectedStart -or [Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -gt 3) { continue }
            } catch { continue }
            try { $actualPath = [string]$process.Path } catch { $actualPath = "" }
            if (-not [string]::IsNullOrWhiteSpace($actualPath) -and
                -not $actualPath.Equals($ExpectedQQPath, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            $validQQ += $process
        }
        if ($validQQ.Count -eq 0) { return $false }
        $listenerPids = @()
        foreach ($line in @(netstat -ano -p tcp 2>$null)) {
            if ($line -match '^\s*TCP\s+127\.0\.0\.1:(5700|6700)\s+\S+\s+LISTENING\s+(\d+)\s*$') {
                $listenerPids += [int]$Matches[2]
            }
        }
        if ($listenerPids.Count -ne 2 -or ($listenerPids | Select-Object -Unique).Count -ne 1 -or
            $validQQ.Id -notcontains [int]$listenerPids[0]) { return $false }
        try { $launcherStart = [DateTimeOffset]$launchers[0].StartTime.ToUniversalTime() } catch { return $false }
        $script:NapCatLauncherPid = [int]$launchers[0].Id
        $script:NapCatLauncherStart = $launcherStart.ToString("o")
        $evidence.launcher_pid = [int]$script:NapCatLauncherPid
        $evidence.launcher_start_time = [string]$script:NapCatLauncherStart
        $evidence.recorded_at_utc = [DateTime]::UtcNow.ToString("o")
        $tmp = "$EvidencePath.tmp"
        $evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding UTF8
        Move-Item -LiteralPath $tmp -Destination $EvidencePath -Force
        Write-StartLog "recovered current NapCat launcher metadata from bounded recorded process evidence" | Out-Null
        return $true
    } catch { return $false }
}

function Test-RootNapCatProcess([string]$Root, [string]$ExecutablePath, [string]$EvidencePath = "") {
    # Process command lines are not exposed for every QQ child on Windows.
    # Prefer an exact configured executable path, then accept the isolated root
    # or profile only when CIM exposes those fields.
    $rootNeedle = $Root.TrimEnd("\")
    $profileNeedle = (Join-Path $rootNeedle "profile").TrimEnd("\")
    $qqPath = if ([string]::IsNullOrWhiteSpace($ExecutablePath)) {
        [IO.Path]::GetFullPath((Join-Path $rootNeedle "qq\QQ.exe"))
    } else {
        [IO.Path]::GetFullPath($ExecutablePath)
    }
    try {
        $processes = Get-CimInstance Win32_Process -ErrorAction Stop
        $launcherPath = Join-Path $rootNeedle "napcat\NapCatWinBootMain.exe"
        $launcherIds = @{}
        foreach ($candidate in @($processes)) {
            if ([string]$candidate.Name -eq "NapCatWinBootMain.exe" -and
                ([string]$candidate.ExecutablePath).Equals($launcherPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                $launcherIds[[int]$candidate.ProcessId] = $true
            }
        }
        $found = $false
        foreach ($process in $processes) {
            $processName = [string]$process.Name
            $path = [string]$process.ExecutablePath
            $commandLine = [string]$process.CommandLine
            if (-not $processName.Equals("QQ.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            $pathKnown = -not [string]::IsNullOrWhiteSpace($path)
            $pathMatches = $pathKnown -and $path.Equals($qqPath, [System.StringComparison]::OrdinalIgnoreCase)
            $profileMatches = $commandLine.IndexOf($profileNeedle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
            $launcherOwned = $launcherIds.ContainsKey([int]$process.ParentProcessId)
            # The NapCat hook and ordinary QQ use the same QQ.exe binary.  A
            # path-only match would mistake the user's default QQ profile for
            # the isolated profile and block a safe startup.  NapCat's QQ
            # root may omit --user-data-dir after bootstrap, so its exact
            # launcher parent is an equally strong ownership fact.  This proves
            # ownership only: it is deliberately not evidence that the
            # requested profile is in use, which is recorded separately by
            # Write-NapCatProcessEvidence and never gates startup.
            if (($pathMatches -or -not $pathKnown) -and ($profileMatches -or $launcherOwned)) {
                $found = $true
                break
            }
        }
        if ($found) {
            return $true
        }
    } catch {
        # Some Windows sessions deny CIM process inspection.  Fall back to
        # the exact executable path exposed by Get-Process; never accept a
        # QQ process from another installation or a registry-discovered path.
    }
    try {
        # Do not use a path-only Get-Process fallback: ordinary QQ can share
        # the configured executable.  If CIM hides both command lines and
        # profile paths, the bounded launch evidence below is the only safe
        # fallback.
        $null = @(Get-Process -Name "QQ" -ErrorAction SilentlyContinue)
    } catch {
        # Fall through to the bounded launch evidence below.
    }
    if (-not [string]::IsNullOrWhiteSpace($EvidencePath)) {
        return Test-RecordedNapCatEvidence $EvidencePath $qqPath $profileNeedle (Join-Path $rootNeedle "napcat\NapCatWinBootMain.exe")
    }
    return $false
}

function Test-ProductionLogin([uri]$HttpUri, [string]$AccessToken) {
    if ([string]::IsNullOrWhiteSpace($AccessToken)) {
        Stop-Production "NAPCAT_ACCESS_TOKEN is missing"
    }
    $base = $HttpUri.AbsoluteUri.TrimEnd("/")
    try {
        $response = Invoke-RestMethod -Method Post -Uri ($base + "/get_login_info") `
            -Headers @{ Authorization = ("Bearer " + $AccessToken) } `
            -ContentType "application/json" -Body "{}" -TimeoutSec 5
        $data = $response.data
        $userId = if ($null -ne $data) { [string]$data.user_id } else { "" }
        return ($response.status -eq "ok" -and [int]$response.retcode -eq 0 -and $userId -match "^[0-9]+$")
    } catch {
        return $false
    }
}

function Get-TargetQQProcesses([string]$ExecutablePath, [string]$ProfilePath = "", [string]$EvidencePath = "") {
    $target = [IO.Path]::GetFullPath($ExecutablePath)
    $profileNeedle = if ([string]::IsNullOrWhiteSpace($ProfilePath)) { "" } else { (Resolve-Path $ProfilePath -ErrorAction SilentlyContinue).Path.TrimEnd("\") }
    try {
        $cimProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        $launcherPath = if ([string]::IsNullOrWhiteSpace($ProfilePath)) { "" } else { Join-Path (Split-Path -Parent $ProfilePath) "napcat\NapCatWinBootMain.exe" }
        $launcherPid = [int]$script:NapCatLauncherPid
        $ownedIds = @{}
        $eligibleIds = @{}
        foreach ($process in $cimProcesses) {
            if (-not ([string]$process.Name).Equals("QQ.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            $path = [string]$process.ExecutablePath
            $commandLine = [string]$process.CommandLine
            $pathKnown = -not [string]::IsNullOrWhiteSpace($path)
            $pathMatches = $pathKnown -and $path.Equals($target, [System.StringComparison]::OrdinalIgnoreCase)
            $profileMatches = -not [string]::IsNullOrWhiteSpace($profileNeedle) -and
                $commandLine.IndexOf($profileNeedle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
            $launcherOwned = $launcherPid -gt 0 -and [int]$process.ParentProcessId -eq $launcherPid
            if (($pathMatches -or -not $pathKnown) -and ($profileMatches -or $launcherOwned)) {
                $ownedIds[[int]$process.ProcessId] = $true
            }
        }
        # Once the NapCat root is identified, include its QQ children even
        # when QQ strips the profile argument from their command lines.
        $changed = $true
        while ($changed) {
            $changed = $false
            foreach ($process in $cimProcesses) {
                if (-not ([string]$process.Name).Equals("QQ.exe", [System.StringComparison]::OrdinalIgnoreCase)) { continue }
                if (-not $ownedIds.ContainsKey([int]$process.ProcessId) -and
                    $ownedIds.ContainsKey([int]$process.ParentProcessId) -and
                    (
                        [string]::IsNullOrWhiteSpace([string]$process.ExecutablePath) -or
                        ([string]$process.ExecutablePath).Equals($target, [System.StringComparison]::OrdinalIgnoreCase)
                    )) {
                    $ownedIds[[int]$process.ProcessId] = $true
                    $changed = $true
                }
            }
        }
        foreach ($id in $ownedIds.Keys) { $eligibleIds[[int]$id] = $true }
        return @(Get-Process -Name "QQ" -ErrorAction SilentlyContinue | Where-Object {
            $eligibleIds.ContainsKey([int]$_.Id)
        })
    } catch {
        # Without a profile-bearing command line, returning every QQ process
        # would make the default QQ profile an unsafe stop/start target.  Use the
        # bounded recorded generation only when the caller supplied evidence.
        if (-not [string]::IsNullOrWhiteSpace($EvidencePath) -and
            (Test-RecordedNapCatEvidence $EvidencePath $target $profileNeedle (Join-Path (Split-Path -Parent $ProfilePath) "napcat\NapCatWinBootMain.exe"))) {
            try {
                $evidence = (Get-Content -Raw -LiteralPath $EvidencePath | ConvertFrom-Json)
                $ids = @($evidence.processes | ForEach-Object { [int]$_.pid })
                return @(Get-Process -Name "QQ" -ErrorAction SilentlyContinue | Where-Object { $ids -contains [int]$_.Id })
            } catch { }
        }
        return @()
    }
}

function Get-ObservedUserDataDir([int]$ProcessId) {
    # Record what a QQ process actually declares instead of what we asked for.
    # Windows can hide CommandLine for protected children; "" means unobserved.
    if ($ProcessId -le 0) { return "" }
    try {
        $process = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $ProcessId) -ErrorAction Stop
    } catch {
        return ""
    }
    if ($null -eq $process) { return "" }
    $commandLine = [string]$process.CommandLine
    if ([string]::IsNullOrWhiteSpace($commandLine)) { return "" }
    $match = [regex]::Match(
        $commandLine,
        '--user-data-dir=(?:"([^"]*)"|([^s"]+))',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $match.Success) { return "" }
    $value = if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
    return $value.TrimEnd("")
}

function Write-NapCatProcessEvidence([string]$RuntimePath, [string]$ExecutablePath, [string]$ProfilePath, [string]$LauncherPath) {
    $processes = Get-TargetQQProcesses $ExecutablePath $ProfilePath $RuntimePath
    $records = @()
    $evidenceLauncherPid = [int]$script:NapCatLauncherPid
    $evidenceLauncherStart = [string]$script:NapCatLauncherStart
    foreach ($process in $processes) {
        $startTime = ""
        try {
            $startTime = $process.StartTime.ToUniversalTime().ToString("o")
        } catch {
            # StartTime can be unavailable for a protected child; PID and
            # executable path still provide a bounded stop target.
        }
        $records += [ordered]@{
            pid = [int]$process.Id
            start_time = $startTime
            user_data_dir = (Get-ObservedUserDataDir ([int]$process.Id))
        }
    }
    # Protected QQ children can disappear from Get-Process.Path after the
    # initial launch.  Do not erase a still-valid launch generation merely
    # because a later evidence refresh cannot read those paths.  The bounded
    # verifier will re-check every retained PID and UTC start time.
    if ($records.Count -eq 0 -and (Test-Path -LiteralPath $RuntimePath -PathType Leaf)) {
        try {
            $previousRaw = Get-Content -Raw -LiteralPath $RuntimePath
            try {
                $previous = $previousRaw | ConvertFrom-Json -DateKind String -ErrorAction Stop
            } catch {
                $previous = $previousRaw | ConvertFrom-Json -ErrorAction Stop
            }
            # Never carry records across a recorded launch generation: stale
            # PIDs must not make a fresh endpoint handoff look healthy.
            $sameGeneration = if ([int]$script:NapCatLauncherPid -gt 0) {
                [int]$previous.launcher_pid -eq [int]$script:NapCatLauncherPid -and
                    ([string]$previous.launcher_start_time -eq [string]$script:NapCatLauncherStart)
            } else {
                Test-RecordedNapCatEvidence $RuntimePath $ExecutablePath $ProfilePath $LauncherPath
            }
            if ($sameGeneration -and
                [string]$previous.schema -eq "qichi-napcat-processes" -and
                ([string]$previous.executable).Equals($ExecutablePath, [System.StringComparison]::OrdinalIgnoreCase) -and
                ([string]$previous.profile).TrimEnd("\").Equals($ProfilePath.TrimEnd("\"), [System.StringComparison]::OrdinalIgnoreCase)) {
                if ($evidenceLauncherPid -le 0) {
                    $evidenceLauncherPid = [int]$previous.launcher_pid
                    $evidenceLauncherStart = [string]$previous.launcher_start_time
                }
                foreach ($record in @($previous.processes)) {
                    if ([int]$record.pid -gt 0 -and -not [string]::IsNullOrWhiteSpace([string]$record.start_time)) {
                        $records += [ordered]@{
                            pid = [int]$record.pid
                            start_time = [string]$record.start_time
                            user_data_dir = [string]$record.user_data_dir
                        }
                    }
                }
            }
        } catch {
            # A corrupt previous evidence file must not block writing a fresh
            # one; the verifier will fail closed until new evidence appears.
        }
    }
    # Separate the launch request from the observation.  "profile" stays the
    # requested path so that existing stop/recovery evidence keeps working;
    # "profile_evidence" states what the running QQ actually declared:
    # matches | differs | unverified.  It is recorded, never used as a gate,
    # because a gate that cannot be satisfied would block a working stack.
    $observedValues = @(
        $records |
            ForEach-Object { [string]$_.user_data_dir } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )
    $observedProfile = ""
    $profileEvidence = "unverified"
    if ($observedValues.Count -eq 1) {
        $observedProfile = [string]$observedValues[0]
        if ($observedProfile.TrimEnd("").Equals($ProfilePath.TrimEnd(""), [System.StringComparison]::OrdinalIgnoreCase)) {
            $profileEvidence = "matches"
        } else {
            $profileEvidence = "differs"
        }
    } elseif ($observedValues.Count -gt 1) {
        $observedProfile = ($observedValues -join "; ")
        $profileEvidence = "differs"
    }
    if ($profileEvidence -ne "matches") {
        Write-StartLog ("NapCat profile evidence {0}: requested '{1}', observed '{2}'" -f $profileEvidence, $ProfilePath, $observedProfile) | Out-Null
    }
    $payload = [ordered]@{
        schema = "qichi-napcat-processes"
        executable = $ExecutablePath
        profile = $ProfilePath
        profile_observed = $observedProfile
        profile_evidence = $profileEvidence
        recorded_at_utc = [DateTime]::UtcNow.ToString("o")
        processes = $records
        launcher_executable = $LauncherPath
        launcher_pid = $evidenceLauncherPid
        launcher_start_time = $evidenceLauncherStart
    }
    try {
        $tmp = "$RuntimePath.tmp"
        $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding UTF8
        Move-Item -LiteralPath $tmp -Destination $RuntimePath -Force
    } catch {
        Write-StartLog "WARN NapCat process evidence could not be written"
    }
}

function Test-RecordedLauncherAlive([int]$ProcessId, [string]$ExpectedPath) {
    if ($ProcessId -le 0) {
        return $false
    }
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $process -or $process.HasExited) {
            return $false
        }
        $expectedName = [IO.Path]::GetFileNameWithoutExtension($ExpectedPath)
        if (-not [string]::IsNullOrWhiteSpace($expectedName) -and
            -not ([string]$process.ProcessName).Equals($expectedName, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        $actualPath = [string]$process.Path
        return ([string]::IsNullOrWhiteSpace($actualPath) -or
            $actualPath.Equals($ExpectedPath, [System.StringComparison]::OrdinalIgnoreCase))
    } catch {
        return $false
    }
}

function Stop-RecordedLauncher([int]$ProcessId, [string]$ExpectedPath) {
    if (-not (Test-RecordedLauncherAlive $ProcessId $ExpectedPath)) {
        return
    }
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
        Write-StartLog "WARN failed to stop the recorded NapCat launcher"
    }
}

function Start-LocalDashboard([string]$ProjectRoot, [string]$PythonPath) {
    # Dashboard is an optional read-only sidecar. Its failure must never block
    # the production bot handoff or expose response content in startup logs.
    $dashboardScript = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "scripts\dashboard_server.py"))
    $dashboardNeedle = $dashboardScript.Replace("'", "''")
    try {
        $processes = Get-CimInstance Win32_Process -ErrorAction Stop
        foreach ($process in $processes) {
            $commandLine = [string]$process.CommandLine
            if ($commandLine.IndexOf($dashboardScript, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $commandLine -match "(?i)(^|\s)--port\s+8765(\s|$)") {
                Write-StartLog "dashboard already running for this project"
                return
            }
        }
    } catch {
        Write-StartLog "WARN dashboard process inspection unavailable"
    }
    if (Test-TcpPort "127.0.0.1" 8765) {
        Write-StartLog "WARN dashboard port occupied by non-project process"
        return
    }
    $stdout = Join-Path $ProjectRoot "runtime\dashboard.stdout.log"
    $stderr = Join-Path $ProjectRoot "runtime\dashboard.stderr.log"
    try {
        Start-Process -FilePath $PythonPath `
            -ArgumentList @($dashboardScript, "--port", "8765") `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr | Out-Null
        Write-StartLog "dashboard started on loopback port 8765"
    } catch {
        Write-StartLog "WARN dashboard start failed; continuing core handoff"
    }
}

Import-PersistedEnvironment

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:StartLogPath = Join-Path $projectRoot "runtime\start-production.log"
try {
    New-Item -ItemType Directory -Path (Split-Path -Parent $script:StartLogPath) -Force -ErrorAction Stop | Out-Null
} catch {
    $script:StartLogPath = $null
}
Write-StartLog "entry"

$napRoot = if ([string]::IsNullOrWhiteSpace($NapCatRoot)) { $env:QICHI_NAPCAT_ROOT } else { $NapCatRoot }
if ([string]::IsNullOrWhiteSpace($napRoot)) {
    $napRoot = "E:\NapCatQQ"
}
try {
    $napRoot = (Resolve-Path $napRoot -ErrorAction Stop).Path
} catch {
    Stop-Production "NapCat root does not exist"
}
Write-StartLog "using configured NapCat root"

$qqExeValue = if (-not [string]::IsNullOrWhiteSpace($QQExecutable)) {
    $QQExecutable
} elseif (-not [string]::IsNullOrWhiteSpace($env:QICHI_QQ_EXE)) {
    $env:QICHI_QQ_EXE
} elseif (Test-Path -LiteralPath "E:\QQ\QQ.exe" -PathType Leaf) {
    # The QQ binary in the NapCat bundle retains the machine install root in
    # its bootstrap metadata.  Use the explicit installed binary while still
    # keeping the NapCat hook and the requested profile under the isolated root.
    "E:\QQ\QQ.exe"
} else {
    Join-Path $napRoot "qq\QQ.exe"
}
try {
    $qqExe = (Resolve-Path $qqExeValue -ErrorAction Stop).Path
} catch {
    Stop-Production "configured QQ executable does not exist"
}
Write-StartLog "using configured QQ executable"
$hook = Join-Path $napRoot "napcat\NapCatWinBootHook.dll"
$launcher = Join-Path $napRoot "napcat\NapCatWinBootMain.exe"
$profile = Join-Path $napRoot "profile"
$napLaunch = Join-Path $napRoot "launch-napcat.ps1"
foreach ($required in @($qqExe, $hook, $launcher, $napLaunch)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Stop-Production "NapCat installation is incomplete"
    }
}
if (-not (Test-Path -LiteralPath $profile -PathType Container)) {
    New-Item -ItemType Directory -Path $profile -Force | Out-Null
}
$patchPackage = Write-NapCatPatchPackage $qqExe (Join-Path $projectRoot "runtime\napcat-qqnt.json")
$quickLoginAccount = Get-NapCatQuickLoginAccount $napRoot
Write-StartLog ("using NapCat quick-login account {0}" -f $quickLoginAccount)

$httpUri = Get-EndpointUri $env:NAPCAT_HTTP_URL "http://127.0.0.1:5700" "http"
$wsUri = Get-EndpointUri $env:NAPCAT_WS_URL "ws://127.0.0.1:6700" "ws"
$accessToken = [string]$env:NAPCAT_ACCESS_TOKEN

$httpReady = Test-TcpPort $httpUri.Host $httpUri.Port
$wsReady = Test-TcpPort $wsUri.Host $wsUri.Port
$napcatEvidencePath = Join-Path $projectRoot "runtime\\napcat-processes.json"
$expectedLauncherPath = Join-Path $napRoot "napcat\NapCatWinBootMain.exe"
if ($httpReady -and $wsReady -and [int]$script:NapCatLauncherPid -le 0) {
    [void](Recover-ExistingNapCatEvidence $napcatEvidencePath $qqExe (Join-Path $napRoot "profile") $expectedLauncherPath)
}
$recordedNapcatProcess = Test-RootNapCatProcess $napRoot $qqExe $napcatEvidencePath
$napcatProcess = if ($httpReady -and $wsReady) { $recordedNapcatProcess } else { $false }
if (-not ($httpReady -and $wsReady) -and $recordedNapcatProcess) {
    Stop-Production "old NapCat process remains while endpoints are down"
}
if ($httpReady -and $wsReady -and -not $napcatProcess) {
    Stop-Production "NapCat endpoints are listening but no process from the configured isolated root was found"
}

$lastLaunchUtc = [DateTime]::MinValue
$lastPhase = ""
$restartCount = 0
$startNapCat = {
    try {
        $env:NAPCAT_QUICK_ACCOUNT = $quickLoginAccount
        $env:NAPCAT_PATCH_PACKAGE = $patchPackage
        $env:NAPCAT_LOAD_PATH = Join-Path $napRoot "loadNapCat.js"
        $env:NAPCAT_INJECT_PATH = $hook
        $env:NAPCAT_LAUNCHER_PATH = $launcher
        $env:NAPCAT_MAIN_PATH = Join-Path $napRoot "napcat\napcat.mjs"
        $env:NAPCAT_WORKDIR = Join-Path $napRoot "napcat"
        $launcherProcess = Start-Process -FilePath $launcher `
            -ArgumentList @($qqExe, $hook, "--user-data-dir=$profile", "-q", $quickLoginAccount) `
            -WorkingDirectory (Join-Path $napRoot "napcat") -WindowStyle Hidden -PassThru
        $script:NapCatLauncherPid = [int]$launcherProcess.Id
        try {
            $script:NapCatLauncherStart = $launcherProcess.StartTime.ToUniversalTime().ToString("o")
        } catch {
            $script:NapCatLauncherStart = ""
        }
        Write-NapCatProcessEvidence $napcatEvidencePath $qqExe $profile $launcher
        $script:lastLaunchUtc = [DateTime]::UtcNow
        $script:restartCount += 1
        return $true
    } catch {
        Stop-Production "isolated NapCat could not be started"
    }
}

if (-not ($httpReady -and $wsReady) -and -not $napcatProcess) {
    if ((Get-TargetQQProcesses $qqExe $profile $napcatEvidencePath).Count -gt 0) {
        Stop-Production "target QQ executable is already running; close it before isolated startup"
    }
    Write-StartLog "starting isolated NapCat"
    & $startNapCat
}

$deadline = [DateTime]::UtcNow.AddSeconds([Math]::Max(10, $LoginWaitSeconds))
$loggedIn = $false
do {
    $httpReady = Test-TcpPort $httpUri.Host $httpUri.Port
    $wsReady = Test-TcpPort $wsUri.Host $wsUri.Port
    $napcatProcess = if ($httpReady -and $wsReady) { Test-RootNapCatProcess $napRoot $qqExe $napcatEvidencePath } else { $false }
    Write-NapCatProcessEvidence $napcatEvidencePath $qqExe $profile $launcher
    if ($httpReady -and $wsReady -and -not $napcatProcess) {
        Stop-Production "NapCat endpoints are listening but the isolated process exited"
    }
    if ($httpReady -and $wsReady -and $napcatProcess) {
        $loggedIn = Test-ProductionLogin $httpUri $accessToken
        if ($loggedIn) {
            if ($lastPhase -ne "logged-in") {
                Write-StartLog "NapCat endpoints and QQ login are ready"
                $lastPhase = "logged-in"
            }
            break
        }
        if ($lastPhase -ne "waiting-login") {
            Write-StartLog "NapCat endpoints are ready; waiting for QQ login"
            $lastPhase = "waiting-login"
        }
    } elseif (Test-RecordedLauncherAlive $script:NapCatLauncherPid $launcher) {
        if ($lastPhase -ne "waiting-endpoints") {
            Write-StartLog "waiting for NapCat HTTP and Forward WS endpoints"
            $lastPhase = "waiting-endpoints"
        }
    } elseif (([DateTime]::UtcNow - $lastLaunchUtc).TotalSeconds -ge 10) {
        if ($restartCount -ge [Math]::Max(1, $MaxNapCatRestarts)) {
            Stop-Production ("NapCat exited repeatedly; restart limit {0} reached" -f [Math]::Max(1, $MaxNapCatRestarts))
        }
        if (Test-RecordedLauncherAlive $script:NapCatLauncherPid $launcher) {
            Write-StartLog "NapCat launcher stayed alive but target QQ did not start; stopping recorded launcher"
            Stop-RecordedLauncher $script:NapCatLauncherPid $launcher
        } else {
            Write-StartLog "NapCat launcher exited while waiting; restarting isolated NapCat"
        }
        & $startNapCat
        $lastPhase = "restarting"
    }
    if ([DateTime]::UtcNow -ge $deadline) {
        break
    }
    Start-Sleep -Seconds ([Math]::Max(1, $PollSeconds))
} while ($true)

if (-not $loggedIn) {
    Stop-Production ("NapCat login/endpoint readiness timed out after {0} seconds" -f [Math]::Max(10, $LoginWaitSeconds))
}
if (-not ($httpReady -and $wsReady -and (Test-RootNapCatProcess $napRoot $qqExe $napcatEvidencePath))) {
    Stop-Production "NapCat process or endpoints became unavailable before Qichi handoff"
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$stack = Join-Path $projectRoot "scripts\start_stack.py"
$config = Join-Path $projectRoot "config.example.yaml"
# 2026-09-14: the primary model was switched to pro while this path stayed hardcoded to
# flash, so a start_bot.bat restart would abort with ModelCapabilityError at handoff.
# Derive the evidence file from the configured primary model instead.
$primaryModel = (& $python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['llm']['primary']['model'])" $config 2>$null | Select-Object -First 1)
if ($primaryModel) { $primaryModel = ([string]$primaryModel).Trim() }
if (-not $primaryModel) { $primaryModel = "deepseek-v4-pro" }
$evidence = Join-Path $projectRoot ("runtime\{0}-capability.json" -f $primaryModel)
Write-StartLog ("Qichi primary model {0}; capability evidence {1}" -f $primaryModel, $evidence)
$ready = Join-Path $projectRoot "runtime\qichi-ready.json"
$lock = Join-Path $projectRoot "runtime\qichi.lock"
foreach ($required in @($python, $stack, $config, $evidence)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Stop-Production "Qichi production files are missing"
    }
}

Write-StartLog "NapCat ready; handing off to Qichi production stack"
Write-NapCatProcessEvidence $napcatEvidencePath $qqExe $profile $launcher
Start-LocalDashboard $projectRoot $python
& $python $stack --config $config --evidence $evidence --ready $ready --lock $lock
$exitCode = $LASTEXITCODE
Write-StartLog ("Qichi production stack exited with code {0}" -f $exitCode)
exit $exitCode
