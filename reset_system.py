#!/usr/bin/env python3
"""
Reset AI Visitor Tracker to a clean slate.

Stops tracker/dashboard processes for this project, deletes visitor_tracking.db,
and removes all cropped images in captures/.

  python reset_system.py          # asks for confirmation
  python reset_system.py --yes    # no prompt (for scripts)
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "visitor_tracking.db"
CAPTURES_DIR = PROJECT_DIR / "captures"
UPLOADS_DIR = PROJECT_DIR / "uploads"

TRACKER_MARKERS = ("main.py", "dashboard.py", "run_system.py")


def _project_path() -> str:
    return str(PROJECT_DIR.resolve()).lower()


def _is_tracker_process(command_line: str) -> bool:
    cmd = command_line.lower()
    if _project_path() not in cmd.replace("/", "\\"):
        return False
    return any(marker in cmd for marker in TRACKER_MARKERS)


def _list_python_processes_win() -> List[Tuple[int, str]]:
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    processes: List[Tuple[int, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0].strip())
        except ValueError:
            continue
        processes.append((pid, parts[1]))
    return processes


def _terminate_pid(pid: int) -> bool:
    if pid == 0:
        return False
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        import os
        import signal

        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def stop_tracking_processes() -> int:
    """Stop python processes running this project's tracker or dashboard."""
    my_pid = 0
    try:
        my_pid = __import__("os").getpid()
    except Exception:
        pass

    stopped = 0
    if sys.platform == "win32":
        for pid, cmdline in _list_python_processes_win():
            if pid == my_pid or not _is_tracker_process(cmdline):
                continue
            print(f"[RESET] Stopping PID {pid} …")
            if _terminate_pid(pid):
                stopped += 1
    else:
        print("[RESET] Auto-stop is supported on Windows only.")
        print("[RESET] Stop run_system.py manually (Ctrl+C) if it is running.")

    if stopped:
        time.sleep(1.5)
    return stopped


def clear_directory(folder: Path, label: str) -> int:
    if not folder.is_dir():
        return 0
    removed = 0
    for item in folder.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
            removed += 1
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    print(f"[RESET] Cleared {removed} item(s) from {label}/")
    return removed


def reset_data() -> None:
    if DB_PATH.is_file():
        DB_PATH.unlink()
        print(f"[RESET] Deleted {DB_PATH.name}")
    else:
        print(f"[RESET] No database file ({DB_PATH.name})")

    CAPTURES_DIR.mkdir(exist_ok=True)
    clear_directory(CAPTURES_DIR, "captures")

    if UPLOADS_DIR.is_dir():
        clear_directory(UPLOADS_DIR, "uploads")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop tracking and wipe visitor database + capture images."
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    print("AI Visitor Tracker — full reset")
    print(f"Project: {PROJECT_DIR}\n")
    print("This will:")
    print("  • Stop run_system.py / main.py / dashboard.py for this project")
    print(f"  • Delete {DB_PATH.name} (all visit + identity records)")
    print("  • Delete all files in captures/ (CCTV thumbnails)")
    print("  • Delete search uploads in uploads/\n")

    if not args.yes:
        answer = input("Type yes to continue: ").strip().lower()
        if answer not in ("yes", "y"):
            print("Cancelled.")
            return 0

    n = stop_tracking_processes()
    print(f"[RESET] Stopped {n} process(es).")
    reset_data()
    print("\n[RESET] Done. Fresh slate ready.")
    print("Next steps:")
    print("  1. python download_face_models.py --no-proxy")
    print("  2. python download_body_model.py --no-proxy")
    print("  3. python download_model.py --no-proxy")
    print("  4. python run_system.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
