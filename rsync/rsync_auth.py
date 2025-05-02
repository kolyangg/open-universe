import configparser, getpass, json, os, pathlib, subprocess, sys
from typing import Optional

CONFIG_DIR  = pathlib.Path.home() / ".config" / "rclone"
RCLONE_CONF = CONFIG_DIR / "rclone.conf"
DROP_REMOTE = "dropbox"
GS_PROFILE  = "gs_service_acc"                 # just a filename for stored key

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _dropbox_cfg_exists() -> bool:
    if not RCLONE_CONF.exists():
        return False
    cp = configparser.ConfigParser()
    cp.read(RCLONE_CONF)
    return DROP_REMOTE in cp and "token" in cp[DROP_REMOTE]

def _write_dropbox_token(access_token: str):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cp = configparser.ConfigParser()
    if RCLONE_CONF.exists():
        cp.read(RCLONE_CONF)

    cp[DROP_REMOTE] = {
        "type":  "dropbox",
        "token": json.dumps(
            {
                "access_token": access_token,
                "token_type":   "bearer",
                "expiry":       "0001-01-01T00:00:00Z",
            }
        ),
    }
    with RCLONE_CONF.open("w") as fh:
        cp.write(fh)
    print("✅ Dropbox token saved to rclone.conf")

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def ensure_login(remote_root: str):
    if remote_root.startswith("dropbox:"):
        _ensure_dropbox()
    elif remote_root.startswith("gs://"):
        _ensure_gs()
    else:        # local path / ssh
        return

# ──────────────────────────────────────────────────────────────
# Dropbox
# ──────────────────────────────────────────────────────────────
def _ensure_dropbox():
    if _dropbox_cfg_exists():
        return                                   # already configured

    print("🛈 No Dropbox credentials found for rclone.")

    if input("Do you already have a Dropbox **access‑token** to paste? [y/N] ").lower().startswith("y"):
        token = getpass.getpass("🔐  Paste access‑token (input hidden): ").strip()
        if not token:
            print("❌ Empty token – aborting.")
            sys.exit(1)
        _write_dropbox_token(token)
        return

    # Fallback – run the full rclone wizard
    print("Launching rclone config wizard …")
    subprocess.run(["rclone", "config"], check=True)

# ──────────────────────────────────────────────────────────────
# Google‑Cloud  (unchanged, just tidied a little)
# ──────────────────────────────────────────────────────────────
def _ensure_gs():
    try:                                         # already authed?
        subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return
    except subprocess.CalledProcessError:
        pass

    if input("🛈 Paste service‑account **JSON** key now? [y/N] ").lower().startswith("y"):
        print("Enter the JSON key, finish with Ctrl‑D (or empty line):")
        json_key = sys.stdin.read().strip()
        if not json_key:
            print("❌ No JSON received – aborting.")
            sys.exit(1)
        key_path = CONFIG_DIR / f"{GS_PROFILE}.json"
        key_path.write_text(json_key)
    else:
        key_path = pathlib.Path(
            getpass.getpass("Path to service‑account key (.json): ").strip()
        ).expanduser()
        if not key_path.exists():
            print(f"❌ File not found: {key_path}")
            sys.exit(1)

    subprocess.run(
        ["gcloud", "auth", "activate-service-account", "--key-file", str(key_path)],
        check=True,
    )
    print("✅ Google Cloud account activated.")

# ──────────────────────────────────────────────────────────────
# CLI helper – optional
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="One‑shot auth helper for rclone‑based scripts")
    p.add_argument("remote_path", help="dropbox:/… or gs://…")
    args = p.parse_args()
    ensure_login(args.remote_path)
    print("All done ✅")
