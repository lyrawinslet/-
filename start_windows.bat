@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "run.py" %*
  goto finish
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys;sys.exit(0 if sys.maxsize.bit_length()==63 else 1)" >nul 2>nul
  if not errorlevel 1 (
    py -3.12 "run.py" %*
    goto finish
  )
  py -3.11 -c "import sys;sys.exit(0 if sys.maxsize.bit_length()==63 else 1)" >nul 2>nul
  if not errorlevel 1 (
    py -3.11 "run.py" %*
    goto finish
  )
  py -3.13 -c "import sys;sys.exit(0 if sys.maxsize.bit_length()==63 else 1)" >nul 2>nul
  if not errorlevel 1 (
    py -3.13 "run.py" %*
    goto finish
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python "run.py" %*
  goto finish
)
echo Python 3.11-3.13 ^(64-bit^) is required. Install Python 3.12 and retry.
set "STATUS=1"
goto cleanup
:finish
set "STATUS=%ERRORLEVEL%"
:cleanup
popd
if "%~1"=="" pause
exit /b %STATUS%
