$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$apiPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$viteCli = Join-Path $frontendRoot "node_modules\vite\bin\vite.js"
$node = (Get-Command node.exe -ErrorAction Stop).Source

if (-not (Test-Path $apiPython)) {
  throw "Project Python was not found at $apiPython. Create the project .venv first."
}
if (-not (Test-Path $viteCli)) {
  throw "Vite was not found at $viteCli. Run npm.cmd install in frontend first."
}

function Ensure-LocalService {
  param(
    [int]$Port,
    [string]$Name,
    [string]$Executable,
    [string[]]$Arguments,
    [string]$WorkingDirectory
  )

  $logBase = Join-Path $env:TEMP "smart-retail-$Name"
  $stdout = "$logBase.log"
  $stderr = "$logBase-error.log"
  $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue

  if (-not $listener) {
    Start-Process -FilePath $Executable `
      -ArgumentList $Arguments `
      -WorkingDirectory $WorkingDirectory `
      -WindowStyle Hidden `
      -RedirectStandardOutput $stdout `
      -RedirectStandardError $stderr
  }

  $deadline = (Get-Date).AddSeconds(180)
  do {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
      return
    }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $deadline)

  throw "$Name did not start on port $Port. Check $stdout and $stderr."
}

Ensure-LocalService `
  -Port 8000 `
  -Name "api" `
  -Executable $apiPython `
  -Arguments @("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory $projectRoot

Ensure-LocalService `
  -Port 5173 `
  -Name "frontend" `
  -Executable $node `
  -Arguments @($viteCli, "--host", "0.0.0.0", "--port", "5173", "--strictPort") `
  -WorkingDirectory $frontendRoot

$healthDeadline = (Get-Date).AddSeconds(180)
$health = $null
do {
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    if ($health.status -eq "ok") {
      break
    }
  } catch {
    Start-Sleep -Seconds 1
    continue
  }
  Start-Sleep -Seconds 1
} while ((Get-Date) -lt $healthDeadline)
if ($null -eq $health -or $health.status -ne "ok") {
  throw "The API did not become ready after startup pre-warming. Check $env:TEMP\smart-retail-api-error.log."
}

$page = Invoke-WebRequest -Uri "http://127.0.0.1:5173/" -UseBasicParsing -TimeoutSec 10
if ($page.StatusCode -ne 200 -or $page.Content -notmatch "<title>Smart Retail Intelligence</title>") {
  throw "The frontend did not return the expected page."
}

Write-Output "API:      http://127.0.0.1:8000/health"
Write-Output "Frontend: http://127.0.0.1:5173/"
