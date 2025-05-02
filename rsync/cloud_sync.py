import os, subprocess, threading
import pytorch_lightning as pl
from rsync.rsync_auth import ensure_login


def _fire_and_forget(local_dir: str, remote_dir: str):
    """
    Spawn rsync in a daemon thread so the main process never blocks.
    """
    cmd = ["rsync", "-az", "--delete", "--partial", local_dir, remote_dir]
    threading.Thread(
        target=lambda: subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT),
        daemon=True,
    ).start()


class RsyncBackup(pl.Callback):
    """
    ① first call happens immediately after the run folder is created  
    ② subsequent calls are executed after every validation pass
    """
    def __init__(self, run_dir: str, remote_root: str | None):
        super().__init__()
        self.enabled = bool(remote_root)
        if not self.enabled:
            return

        run_dir = os.path.abspath(run_dir)
        self.local_dir = run_dir if run_dir.endswith(os.sep) else run_dir + os.sep
        self.remote_dir = (
            remote_root.rstrip("/\\") + "/" + os.path.basename(run_dir) + "/"
        )

        ensure_login(remote_root)          # one‑time auth
        _fire_and_forget(self.local_dir, self.remote_dir)

    # ‑‑‑ Lightning hooks ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def on_validation_end(self, trainer, pl_module):
        if self.enabled:
            _fire_and_forget(self.local_dir, self.remote_dir)
