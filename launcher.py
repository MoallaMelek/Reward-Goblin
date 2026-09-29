"""Double-click entry point: start the Reward Goblin backend and open the UI in a browser.

    python launcher.py              (from source)
    RewardGoblin.exe                (PyInstaller build, see scripts/build_exe.ps1)

In the packaged app, everything writable (runs, recordings, models, experiments) lives in
the ``data`` folder next to the executable.
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parent


def _free_port(start: int = 8000) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("No free port between 8000 and 8099.")


def main() -> None:
    data = _base_dir()
    os.environ.setdefault("REWARD_GOBLIN_ROOT", str(data))
    os.environ.setdefault("REWARD_GOBLIN_FRONTEND", str(data / "frontend"))

    import uvicorn

    from backend.api.main import app

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("  REWARD GOBLIN - the AI that does exactly what you asked")
    print(f"  UI:   {url}")
    print(f"  Data: {data}")
    print("  Close this window to stop the server.")
    print("=" * 60, flush=True)
    if not os.environ.get("REWARD_GOBLIN_NO_BROWSER"):
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # training workers are spawned processes
    main()
