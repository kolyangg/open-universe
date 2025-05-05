import os
import argparse
import subprocess
from rsync_auth import ensure_login

# 📁 Define folders or files (glob-style) to exclude from upload
EXCLUDE_PATTERNS = ["checkpoints/universe/exper/best-model.ckpt", 
                    "data/",
                    "wandb/latest-run/files/media/**",  # any media artefacts in the latest run
                    "wandb/run-*/files/media/**",       # media artefacts in every run-* folder]
]

def upload_to_cloud(local_exp_path: str, remote_exp_root: str):
    """
    Uploads a local experiment folder to the cloud, maintaining its structure.

    local_exp_path: e.g. exp/gan_run/2025-05-02_15-20-11_lr-3e4
    remote_exp_root: e.g. dropbox:/speech_enh_backups
    """
    if not os.path.isdir(local_exp_path):
        raise ValueError(f"❌ Path does not exist or is not a folder: {local_exp_path}")

    # Normalize paths
    local_exp_path = os.path.abspath(local_exp_path)
    exp_name = os.path.basename(os.path.dirname(local_exp_path))
    run_folder = os.path.basename(local_exp_path)
    remote_path = os.path.join(remote_exp_root.rstrip("/"), exp_name, run_folder)

    ensure_login(remote_exp_root)

    local_dir = local_exp_path.rstrip("/") + "/"
    remote_dir = remote_path.rstrip("/") + "/"

    # Construct rclone command with exclusions
    cmd = ["rclone", "sync", "--progress", 
           "--local-no-check-updated",   # <‑‑ NEW
           local_dir, remote_dir]
    for pattern in EXCLUDE_PATTERNS:
        cmd += ["--exclude", pattern]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ Uploaded to {remote_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local_exp_path", help="Local exp path (e.g. exp/gan_run/2025-...)")
    parser.add_argument("remote_exp_root", help="Cloud base path (e.g. dropbox:/speech_enh_backups)")
    args = parser.parse_args()

    upload_to_cloud(args.local_exp_path, args.remote_exp_root)
