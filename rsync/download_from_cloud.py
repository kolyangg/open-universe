import os
import argparse
import subprocess
from rsync_auth import ensure_login

def download_from_cloud(remote_path: str, local_root: str = "exp"):
    """
    remote_path: full cloud path like `dropbox:/speech_enh_backups/gan_run/2025-05-02_15-20-11_lr-3e4`
    local_root: root folder on disk where the folder will be recreated
    """
    ensure_login(remote_path)

    exp_name = os.path.basename(os.path.dirname(remote_path.rstrip("/")))
    timestamp = os.path.basename(remote_path.rstrip("/"))

    local_path = os.path.join(local_root, exp_name, timestamp)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # rsync always expects trailing slashes for directories
    remote_path = remote_path.rstrip("/") + "/"
    local_path = local_path.rstrip("/") + "/"

    cmd = ["rclone", "sync", "--progress", remote_path, local_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ Downloaded to {local_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_path", help="Full cloud path (e.g. dropbox:/bucket/exp_name/time)")
    args = parser.parse_args()

    download_from_cloud(args.remote_path)
