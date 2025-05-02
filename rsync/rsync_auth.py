import configparser
import getpass
import json
import os
import pathlib
import subprocess
import sys
from typing import Optional

CONFIG_DIR   = pathlib.Path.home() / ".config" / "rclone"
RCLONE_CONF  = CONFIG_DIR / "rclone.conf"
DROP_REMOTE  = "dropbox"          # remote name we’ll look for / create
GS_PROFILE   = "gs_service_acc"   # label for stored key (ours only)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _save_rclone_conf(text: str):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with RCLONE_CONF.open("a") as fh:
        fh.write("\n" + text.strip() + "\n")

def _dropbox_cfg_exists() -> bool:
    if not RCLONE_CONF.exists():
        return False
    cp = configparser.ConfigParser()
    cp.read(RCLONE_CONF)
    return DROP_REMOTE in cp and "token" in cp[DROP_REMOTE]

def _prompt_multiline(msg: str) -> str:
    print(msg)
    print("(finish with an empty line)")
    buf = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        buf.append(line)
    return "\n".join(buf)

# -----------------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------------

def ensure_login(remote_root: str):
    """
    Ensures this process (and future rclone/rclone‑based scripts) are fully
    authenticated for the cloud backend implied by *remote_root*.
    Supported:
        • dropbox:/…   (uses rclone’s Dropbox backend)
        • gs://…       (Google Cloud Storage via gcloud + service‑account)
    """

    if remote_root.startswith("dropbox:"):
        _ensure_dropbox()
    elif remote_root.startswith("gs://"):
        _ensure_gs()
    else:
        # Plain SSH / local path – nothing special
        return


# -----------------------------------------------------------------------------
#  Backend‑specific helpers
# -----------------------------------------------------------------------------

def _ensure_dropbox():
    """
    1) If rclone config already contains a working 'dropbox' remote → done.
    2) Else ask user if they want to paste a config snippet. If yes → save it.
    3) Otherwise run `rclone config` and let the user auth through the browser.
    """
    if _dropbox_cfg_exists():
        return                                                # already ok

    have_cfg = input(
        "🛈 Dropbox remote not configured.\n"
        "Do you already have a dropbox section from rclone.conf to paste? [y/N] "
    ).strip().lower().startswith("y")

    if have_cfg:
        snippet = _prompt_multiline(
            "\nPaste the **[dropbox]** section (including 'token = {...}')"
        )
        if "[dropbox]" not in snippet:
            snippet = f"[dropbox]\n{snippet}"
        _save_rclone_conf(snippet)
        if not _dropbox_cfg_exists():
            print("❌ Could not detect valid token in pasted config. Please try again.")
            sys.exit(1)
        print("✅ Dropbox credentials saved.")
        return

    # Fallback – launch interactive wizard
    print("Launching rclone config wizard …")
    subprocess.run(["rclone", "config"], check=True)


def _ensure_gs():
    """
    1) Checks if gcloud already has an active account with valid ADC.
    2) Otherwise asks if user wants to paste a service‑account JSON key.
    3) Else prompts for a path to a key and activates it via gcloud.
    """
    # Quick test: does 'gcloud auth print-access-token' succeed?
    try:
        subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return                            # already authenticated
    except subprocess.CalledProcessError:
        pass

    paste = input(
        "🛈 Google Cloud auth not found.\n"
        "Do you have a service‑account **JSON** key to paste now? [y/N] "
    ).strip().lower().startswith("y")

    if paste:
        json_key = _prompt_multiline("\nPaste the service‑account JSON:")
        key_path = CONFIG_DIR / f"{GS_PROFILE}.json"
        key_path.write_text(json_key)
    else:
        key_path_str = getpass.getpass(
            "Path to service‑account key (.json): "
        ).strip()
        key_path = pathlib.Path(key_path_str).expanduser().resolve()
        if not key_path.exists():
            print(f"❌ File not found: {key_path}")
            sys.exit(1)

    subprocess.run(
        ["gcloud", "auth", "activate-service-account", "--key-file", str(key_path)],
        check=True,
    )
    print("✅ Google Cloud account activated.")
    
    

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="One‑shot auth helper for rclone‑backed rsync scripts")
    p.add_argument("remote_path",
                   help="dropbox:/… or gs://… — determines which auth flow to run")
    args = p.parse_args()
    ensure_login(args.remote_path)
    print("All done ✅")
