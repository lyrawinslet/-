#!/bin/sh
# macOS: bash start_mac.command; or chmod +x start_mac.command and double-click.
cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
PY=""
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys;sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) and sys.maxsize>2**32 else 1)' >/dev/null 2>&1; then
        PY="$candidate"
        break
      fi
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.11-3.13 (64-bit) is required. Install Python 3.12 and retry."
  exit 1
fi
"$PY" run.py "$@"
status=$?
if [ "$#" -eq 0 ] && [ -t 0 ]; then
  printf '\nPress Enter to close...'
  read answer
fi
exit "$status"
