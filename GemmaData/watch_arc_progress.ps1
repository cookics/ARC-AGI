[CmdletBinding()]
param(
    [string]$RunDir,
    [string]$LogPath,
    [int]$PollSeconds = 5,
    [int]$TailLines = 2,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunsRoot = Join-Path $ScriptRoot "runs"

function Get-JsonSafe {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Format-Span {
    param([Parameter(Mandatory = $true)][TimeSpan]$Span)

    if ($Span.TotalSeconds -lt 0) {
        $Span = [TimeSpan]::Zero
    }
    if ($Span.TotalHours -ge 1) {
        return "{0}h{1:00}m" -f [int]$Span.TotalHours, $Span.Minutes
    }
    if ($Span.TotalMinutes -ge 1) {
        return "{0}m{1:00}s" -f [int]$Span.TotalMinutes, $Span.Seconds
    }
    return "{0}s" -f [int]$Span.TotalSeconds
}

function Get-Bar {
    param(
        [Parameter(Mandatory = $true)][int]$Completed,
        [Parameter(Mandatory = $true)][int]$Total,
        [int]$Width = 24
    )

    if ($Total -le 0) {
        $Total = 1
    }
    $filled = [Math]::Min($Width, [Math]::Max(0, [int][Math]::Round($Width * $Completed / $Total)))
    return "[{0}{1}]" -f ("#" * $filled), ("." * ($Width - $filled))
}

function Resolve-RunDir {
    param([string]$RequestedRunDir)

    if ($RequestedRunDir) {
        return (Resolve-Path -LiteralPath $RequestedRunDir).Path
    }

    if (-not (Test-Path -LiteralPath $RunsRoot)) {
        throw "Runs directory not found: $RunsRoot"
    }

    $candidates = Get-ChildItem -LiteralPath $RunsRoot -Directory | Sort-Object LastWriteTime -Descending
    foreach ($candidate in $candidates) {
        $configPath = Join-Path $candidate.FullName "config.json"
        if (-not (Test-Path -LiteralPath $configPath)) {
            continue
        }

        $config = Get-JsonSafe -Path $configPath
        if ($null -eq $config) {
            continue
        }

        $summaryPath = Join-Path $candidate.FullName "summary.json"
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            return $candidate.FullName
        }

        $requestedTasks = 0
        if ($config.PSObject.Properties.Name -contains "requested_tasks") {
            $requestedTasks = [int]$config.requested_tasks
        } elseif ($config.PSObject.Properties.Name -contains "limit") {
            $requestedTasks = [int]$config.limit
        }

        $tasksDir = Join-Path $candidate.FullName "tasks"
        $completed = if (Test-Path -LiteralPath $tasksDir) {
            (Get-ChildItem -LiteralPath $tasksDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
        } else {
            0
        }

        if ($requestedTasks -gt 0 -and $completed -lt $requestedTasks) {
            return $candidate.FullName
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }

    throw "No run directories found under $RunsRoot"
}

function Resolve-LogPath {
    param([string]$RequestedLogPath)

    if ($RequestedLogPath) {
        return (Resolve-Path -LiteralPath $RequestedLogPath).Path
    }

    $candidates = Get-ChildItem -LiteralPath $ScriptRoot -File -Filter "batch*.out" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }

    return $null
}

function Get-RunnerProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][string]$ScriptDirectory
    )

    $escapedRunDir = [Regex]::Escape($RunDirectory)
    $escapedScriptDir = [Regex]::Escape($ScriptDirectory)
    $escapedRunnerPath = [Regex]::Escape((Join-Path $ScriptDirectory "gemma_arc1_runner.py"))

    $pythonProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    if (-not $pythonProcesses) {
        return @()
    }

    return @(
        $pythonProcesses | Where-Object {
            $cmd = [string]$_.CommandLine
            $cmd -match 'gemma_arc1_runner\.py' -and (
                $cmd -match $escapedRunDir -or
                $cmd -match $escapedRunnerPath -or
                $cmd -match $escapedScriptDir
            )
        }
    )
}

$RunDir = Resolve-RunDir -RequestedRunDir $RunDir
$RunPath = Get-Item -LiteralPath $RunDir
$ConfigPath = Join-Path $RunPath.FullName "config.json"
$ResumePath = Join-Path $RunPath.FullName "resume.json"
$StatusPath = Join-Path $RunPath.FullName "status.json"
$LogPath = Resolve-LogPath -RequestedLogPath $LogPath
$Config = Get-JsonSafe -Path $ConfigPath
$ResumeConfig = if (Test-Path -LiteralPath $ResumePath) { Get-JsonSafe -Path $ResumePath } else { $null }
$RuntimeConfig = if ($ResumeConfig) { $ResumeConfig } else { $Config }
$StartTime = (Get-Item -LiteralPath $ConfigPath).CreationTime

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "model")) {
    $Model = [string]$RuntimeConfig.model
} else {
    $Model = "unknown"
}

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "workers")) {
    $Workers = [int]$RuntimeConfig.workers
} else {
    $Workers = 0
}

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "rate_limit_per_minute")) {
    $RateLimit = [int]$RuntimeConfig.rate_limit_per_minute
} else {
    $RateLimit = 0
}

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "max_output_tokens")) {
    $MaxTokens = [int]$RuntimeConfig.max_output_tokens
} else {
    $MaxTokens = 0
}

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "transient_throttle_per_minute")) {
    $TransientThrottle = [double]$RuntimeConfig.transient_throttle_per_minute
} else {
    $TransientThrottle = 0.0
}

if ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "requested_tasks")) {
    $Total = [int]$RuntimeConfig.requested_tasks
} elseif ($RuntimeConfig -and ($RuntimeConfig.PSObject.Properties.Name -contains "limit")) {
    $Total = [int]$RuntimeConfig.limit
} else {
    $Total = 0
}

while ($true) {
    $runnerProcesses = Get-RunnerProcesses -RunDirectory $RunPath.FullName -ScriptDirectory $ScriptRoot
    $runnerIsLive = $runnerProcesses.Count -gt 0
    $status = if (Test-Path -LiteralPath $StatusPath) {
        Get-JsonSafe -Path $StatusPath
    } else {
        $null
    }

    $tasksDir = Join-Path $RunPath.FullName "tasks"
    $taskFiles = if (Test-Path -LiteralPath $tasksDir) {
        Get-ChildItem -LiteralPath $tasksDir -Filter "*.json" -File -ErrorAction SilentlyContinue
    } else {
        @()
    }

    $completed = $taskFiles.Count
    $solved = 0
    $errors = 0
    $durations = New-Object System.Collections.Generic.List[double]

    foreach ($taskFile in $taskFiles) {
        $task = Get-JsonSafe -Path $taskFile.FullName
        if ($null -eq $task) {
            continue
        }
        if ($task.exact_match -eq $true) {
            $solved++
        }
        if ($task.status -eq "error") {
            $errors++
        }
        if ($task.PSObject.Properties.Name -contains "duration_seconds" -and $null -ne $task.duration_seconds) {
            [void]$durations.Add([double]$task.duration_seconds)
        }
    }

    if ($Total -le 0) {
        $Total = $completed
    }

    $elapsed = (Get-Date) - $StartTime
    $progress = if ($Total -gt 0) { [Math]::Min(1.0, $completed / [double]$Total) } else { 0.0 }
    $ratePerHour = if ($elapsed.TotalHours -gt 0) { $completed / $elapsed.TotalHours } else { 0.0 }
    $ratePerMinute = $ratePerHour / 60.0
    $remaining = [Math]::Max(0, $Total - $completed)
    $eta = if ($ratePerHour -gt 0 -and $remaining -gt 0) {
        [TimeSpan]::FromHours($remaining / $ratePerHour)
    } else {
        [TimeSpan]::Zero
    }

    $accuracy = if ($completed -gt 0) { $solved / [double]$completed } else { 0.0 }
    $avgDuration = if ($durations.Count -gt 0) { ($durations | Measure-Object -Average).Average } else { 0.0 }

    Clear-Host
    Write-Host "ARC batch watcher"
    Write-Host ("Run:    {0}" -f $RunPath.FullName)
    Write-Host ("Model:  {0}" -f $Model)
    if ($TransientThrottle -gt 0) {
        Write-Host ("Config: workers={0} rate={1}/min max_tokens={2} transient_throttle={3}/min" -f `
            $Workers, $RateLimit, $MaxTokens, $TransientThrottle)
    } else {
        Write-Host ("Config: workers={0} rate={1}/min max_tokens={2}" -f $Workers, $RateLimit, $MaxTokens)
    }
    if ($runnerIsLive) {
        $runnerPids = ($runnerProcesses | Select-Object -ExpandProperty ProcessId) -join ", "
        Write-Host ("Runner: live   python_processes={0}   pids={1}" -f $runnerProcesses.Count, $runnerPids)
        Write-Host ("Workers: configured={0}   live_worker_slots~={1}" -f $Workers, $Workers)
    } else {
        Write-Host "Runner: stopped   python_processes=0"
        Write-Host "Workers: configured from resume/config only; live_worker_slots=0"
    }
    Write-Host ("Status: {0}/{1} completed" -f $completed, $Total)
    Write-Host ("Bar:    {0} {1}/{2} ({3:P1})" -f (Get-Bar -Completed $completed -Total $Total), $completed, $Total, $progress)
    Write-Host ("Solved: {0}   Errors: {1}   Accuracy: {2:P1}" -f $solved, $errors, $accuracy)
    if ($status) {
        if ($runnerIsLive) {
            Write-Host ("Requests: started={0} succeeded={1} failed={2} in-flight={3}" -f `
                $status.requests_started, $status.requests_succeeded, $status.requests_failed, $status.requests_in_flight)
        } else {
            Write-Host ("Requests: runner is stopped; last saved snapshot was started={0} succeeded={1} failed={2} in-flight={3}" -f `
                $status.requests_started, $status.requests_succeeded, $status.requests_failed, $status.requests_in_flight)
            Write-Host "Requests: live in-flight count is 0; the saved in-flight value above is stale."
        }
        if ($status.request_control -and $runnerIsLive) {
            $control = $status.request_control
            Write-Host ("Limiter:  base={0:N2}/min current={1:N2}s cooldown={2:N0}s active={3}" -f `
                $control.requests_per_minute_base, $control.current_interval_seconds, $control.cooldown_remaining_seconds, $control.active_requests)
        } elseif ($status.request_control) {
            $control = $status.request_control
            Write-Host ("Limiter:  runner stopped; last saved interval={0:N2}s active={1} (stale snapshot)" -f `
                $control.current_interval_seconds, $control.active_requests)
        }
        if ($status.last_event) {
            $lastEvent = $status.last_event
            $eventBits = @($lastEvent.kind, "task=$($lastEvent.task_id)")
            if ($lastEvent.PSObject.Properties.Name -contains "attempt") {
                $eventBits += "attempt=$($lastEvent.attempt)"
            }
            if ($lastEvent.PSObject.Properties.Name -contains "status" -and $lastEvent.status) {
                $eventBits += "status=$($lastEvent.status)"
            }
            Write-Host ("Last req: {0}" -f ($eventBits -join " "))
        }
        if ($status.PSObject.Properties.Name -contains "status_write_errors") {
            Write-Host ("Status file writes: errors={0}" -f $status.status_write_errors)
            if ($status.last_status_write_error) {
                Write-Host ("Last status write error: {0}" -f $status.last_status_write_error)
            }
        }
        if ($status.PSObject.Properties.Name -contains "stop_requested" -and $status.stop_requested) {
            if ($status.stop_reason) {
                $stopReason = $status.stop_reason
                Write-Host ("Stop requested: reason={0} task={1} attempt={2}" -f `
                    $stopReason.reason, $stopReason.task_id, $stopReason.attempt)
            } else {
                Write-Host "Stop requested: yes"
            }
        }
        if ($status.PSObject.Properties.Name -contains "transient_errors" -and $status.transient_errors) {
            $recentTransientErrors = @($status.transient_errors | Select-Object -Last 8)
            Write-Host ("Recent transient errors: {0}" -f $status.transient_errors.Count)
            foreach ($entry in $recentTransientErrors) {
                $taskId = if ($entry.PSObject.Properties.Name -contains "task_id") { [string]$entry.task_id } else { "unknown" }
                $attempt = if ($entry.PSObject.Properties.Name -contains "attempt" -and $null -ne $entry.attempt) { [string]$entry.attempt } else { "-" }
                $timestamp = if ($entry.PSObject.Properties.Name -contains "timestamp_utc") { [string]$entry.timestamp_utc } else { "" }
                $message = if ($entry.PSObject.Properties.Name -contains "error") { [string]$entry.error } else { "" }
                if ($message.Length -gt 120) {
                    $message = $message.Substring(0, 117) + "..."
                }
                Write-Host ("  503-ish: task={0} attempt={1} time={2}" -f $taskId, $attempt, $timestamp)
                Write-Host ("    {0}" -f $message)
            }
        }
    } else {
        Write-Host "Requests: status.json not present yet"
    }
    Write-Host ("Elapsed: {0}   Rate: {1:N2} tasks/hour ({2:N2} tasks/min)" -f (Format-Span -Span $elapsed), $ratePerHour, $ratePerMinute)
    Write-Host ("ETA:     {0}" -f (Format-Span -Span $eta))
    if ($avgDuration -gt 0) {
        Write-Host ("Avg task duration: {0:N1}s" -f $avgDuration)
    }

    if (Test-Path -LiteralPath $LogPath) {
        $tail = Get-Content -LiteralPath $LogPath -Tail $TailLines -ErrorAction SilentlyContinue
        if ($tail) {
            Write-Host "Latest log:"
            foreach ($line in $tail) {
                Write-Host ("  {0}" -f $line)
            }
        }
    }

    if ($completed -ge $Total -and (Test-Path -LiteralPath (Join-Path $RunPath.FullName "summary.json"))) {
        Write-Host ""
        Write-Host "Run complete."
        break
    }

    if ($Once) {
        break
    }

    Write-Host ""
    Write-Host ("Refreshing every {0}s. Press Ctrl+C to stop." -f $PollSeconds)
    Start-Sleep -Seconds $PollSeconds
}
