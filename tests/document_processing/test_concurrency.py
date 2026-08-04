from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_paddle_slot_uses_one_host_global_root(monkeypatch, tmp_path):
    from document_processing import concurrency

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(concurrency, "get_default_hermes_root", lambda: root)

    with concurrency.paddle_slot("hr-agent", timeout=1.0) as first_wait:
        path = concurrency._paddle_lock_path()
        assert path == root / "cache" / "document-processing" / "locks" / "paddle-0.lock"
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert first_wait >= 0.0

    with concurrency.paddle_slot("purchase-agent", timeout=1.0):
        assert concurrency._paddle_lock_path() == path


def test_paddle_slot_releases_after_context_error(monkeypatch, tmp_path):
    from document_processing import concurrency

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(concurrency, "get_default_hermes_root", lambda: root)

    with pytest.raises(RuntimeError, match="boom"):
        with concurrency.paddle_slot("hr-agent", timeout=1.0):
            raise RuntimeError("boom")

    with concurrency.paddle_slot("purchase-agent", timeout=1.0):
        pass


def test_second_process_times_out_while_paddle_slot_is_held(tmp_path):
    from document_processing import concurrency

    root = tmp_path / ".hermes"
    root.mkdir()
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        "import time\n"
        "from document_processing.concurrency import paddle_slot\n"
        "with paddle_slot('holder-agent', timeout=2.0):\n"
        " print('READY', flush=True)\n"
        " time.sleep(2.0)\n"
    )
    env = {
        **os.environ,
        "HERMES_HOME": str(root),
        "PYTHONPATH": str(repo_root),
    }
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        original = concurrency.get_default_hermes_root
        concurrency.get_default_hermes_root = lambda: root
        try:
            with pytest.raises(concurrency.DocumentConcurrencyError, match="(?i)timed out"):
                with concurrency.paddle_slot("waiting-agent", timeout=0.2):
                    pass
        finally:
            concurrency.get_default_hermes_root = original
    finally:
        child.terminate()
        child.wait(timeout=5)
