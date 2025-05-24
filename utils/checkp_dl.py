#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import zipfile
import re


def install_gdown():
    """Installs gdown via pip in the current environment."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"]);


try:
    import gdown
except ImportError:
    install_gdown()
    import gdown


def extract_file_id(url):
    """Extracts the Google Drive file ID from a shareable URL."""
    patterns = [
        r'/d/([a-zA-Z0-9_-]+)',       # URLs like /d/<id>/
        r'id=([a-zA-Z0-9_-]+)'        # URLs with ?id=<id>
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract file ID from URL: {url}")


def download_and_unzip(url, dest_folder):
    """Downloads a zip from Google Drive and unzips it in the destination folder."""
    os.makedirs(dest_folder, exist_ok=True)
    file_id = extract_file_id(url)
    download_url = f'https://drive.google.com/uc?id={file_id}&export=download'
    output_path = os.path.join(dest_folder, f'{file_id}.zip')

    print(f"Downloading {url} to {output_path}")
    # quiet=False shows a progress bar
    gdown.download(download_url, output=output_path, quiet=False)

    print(f"Unzipping {output_path} to {dest_folder}")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    print(f"Removing archive {output_path}")
    os.remove(output_path)


def main():
    parser = argparse.ArgumentParser(description='Download and unzip a Google Drive file.')
    parser.add_argument('url', help='Google Drive shareable URL')
    parser.add_argument('dest', help='Destination folder')
    args = parser.parse_args()
    download_and_unzip(args.url, args.dest)

if __name__ == '__main__':
    main()