import os, subprocess, threading, time, shlex
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from rsync.rsync_auth import ensure_login

EXCLUDE_PATTERNS = ["checkpoints/universe/exper/best-model.ckpt", "data/"]

# ──────────────────────────────────────────────────────────────
# background worker
# ──────────────────────────────────────────────────────────────
def _sync_worker(local_dir: str, remote_dir: str):
    t0 = time.strftime("%H:%M:%S")
    print(f"[RsyncBackup] ⬆️  {t0}  upload → {remote_dir}")
    cmd = ["rclone", "sync", "--progress", "--stats-one-line", local_dir, remote_dir]
    for pat in EXCLUDE_PATTERNS:
        cmd += ["--exclude", pat]
    rc = subprocess.run(cmd).returncode
    t1 = time.strftime("%H:%M:%S")
    status = "✅ finished" if rc == 0 else f"❌ exit={rc}"
    print(f"[RsyncBackup] {status} {t1} → {remote_dir}")
    if rc:
        print(f"             cmd: {shlex.join(cmd)}")

def _fire_and_forget(local_dir: str, remote_dir: str):
    threading.Thread(target=_sync_worker,
                     args=(local_dir, remote_dir),
                     daemon=True).start()

# ──────────────────────────────────────────────────────────────
# Lightning callback
# ──────────────────────────────────────────────────────────────
class RsyncBackup(pl.Callback):
    """
    • First upload right after Hydra creates the *current* run folder
    • Upload again every time a checkpoint is saved
    """

    def __init__(self, remote_root: str | None):
        super().__init__()
        self.enabled = bool(remote_root)
        if not self.enabled:
            return

        # ------------------------------------------------------------------
        # Local directory: Hydra has already chdir‑ed into the new run dir,
        # irrespective of whether we're resuming or starting fresh.
        # ------------------------------------------------------------------
        self.local_dir = os.path.abspath(os.getcwd()).rstrip(os.sep) + os.sep

        # ------------------------------------------------------------------
        # Remote directory mirrors the relative path of local_dir inside exp/
        # so every (new or resumed) run gets its *own* folder in the cloud.
        # ------------------------------------------------------------------
        project_root = get_original_cwd()
        exp_root     = os.path.join(project_root, "exp")
        rel_path     = os.path.relpath(self.local_dir, exp_root)
        self.remote_dir = (
            remote_root.rstrip("/").rstrip("\\") + "/" + rel_path.rstrip("/\\") + "/"
        )

        print(f"[RsyncBackup] local_dir  = {self.local_dir}")
        print(f"[RsyncBackup] remote_dir = {self.remote_dir}")

        ensure_login(remote_root)
        _fire_and_forget(self.local_dir, self.remote_dir)   # initial sync

    # every time ModelCheckpoint writes a file
    def on_save_checkpoint(self, *_):
        if self.enabled:
            _fire_and_forget(self.local_dir, self.remote_dir)



# import os, subprocess, threading, time, textwrap, shlex
# import pytorch_lightning as pl
# from rsync.rsync_auth import ensure_login

# # Skip these globs
# EXCLUDE_PATTERNS = ["checkpoints/universe/exper/best-model.ckpt"]

# # ──────────────────────────────────────────────────────────────
# # utilities
# # ──────────────────────────────────────────────────────────────
# def _print_snapshot(path: str, max_depth: int = 2, max_files: int = 5):
#     print("[RsyncBackup] folder snapshot:")
#     base = path.rstrip(os.sep) + os.sep
#     for root, dirs, files in os.walk(path):
#         depth = root[len(base):].count(os.sep)
#         if depth > max_depth:
#             continue
#         indent = "  " * depth
#         print(f"{indent}{os.path.basename(root) or '.'}/")
#         for f in files[:max_files]:
#             print(f"{indent}  {f}")
#         if len(files) > max_files:
#             print(f"{indent}  …({len(files)-max_files} more)")
#         dirs[:] = [d for d in dirs if depth < max_depth]  # prune deeper walk

# def _sync_worker(local_dir: str, remote_dir: str):
#     t0 = time.strftime("%H:%M:%S")
#     print(f"[RsyncBackup] ⬆️  {t0}  upload → {remote_dir}")
#     # _print_snapshot(local_dir)

#     cmd = ["rclone", "sync", "--progress", "--stats-one-line", local_dir, remote_dir]
#     for pat in EXCLUDE_PATTERNS:
#         cmd += ["--exclude", pat]

#     # stream output live so progress bar is visible
#     rc = subprocess.run(cmd).returncode
#     t1 = time.strftime("%H:%M:%S")

#     if rc == 0:
#         print(f"[RsyncBackup] ✅ {t1}  finished → {remote_dir}")
#     else:
#         print(f"[RsyncBackup] ❌ {t1}  exit={rc} → {remote_dir}")
#         print(f"             cmd: {shlex.join(cmd)}")

# def _fire_and_forget(local_dir: str, remote_dir: str):
#     threading.Thread(target=_sync_worker,
#                      args=(local_dir, remote_dir),
#                      daemon=True).start()

# # ──────────────────────────────────────────────────────────────
# # Lightning callback
# # ──────────────────────────────────────────────────────────────
# class RsyncBackup(pl.Callback):
#     """
#     • first upload right after the run folder is created
#     • upload after each validation epoch *after* checkpoints are written
#     """

#     def __init__(self, run_dir: str, remote_root: str | None):
#         super().__init__()
#         self.enabled = bool(remote_root)
#         if not self.enabled:
#             return

#         # local dir = Hydra run folder (cwd)
#         self.local_dir = os.path.abspath(os.getcwd()).rstrip(os.sep) + os.sep

#         # remote dir  <remote_root>/<exp_name>/<run_folder>/
#         run_dir_norm = run_dir.rstrip("/\\")
#         exp_name     = os.path.basename(os.path.dirname(run_dir_norm))
#         run_folder   = os.path.basename(run_dir_norm)
#         self.remote_dir = (
#             remote_root.rstrip("/").rstrip("\\")
#             + f"/{exp_name}/{run_folder}/"
#         )

#         ensure_login(remote_root)
#         _fire_and_forget(self.local_dir, self.remote_dir) # initial sync

#    # ⚡ called immediately after ModelCheckpoint (and the module’s own
#     #   on_save_checkpoint) has flushed the .ckpt file to disk
#     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#         if self.enabled:
#             _fire_and_forget(self.local_dir, self.remote_dir)