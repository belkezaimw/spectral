"""
download.py
===========
Download and verify DIV2K + Flickr2K datasets.

Usage:
    uv run gf-download --dest data/ --dataset div2k
    uv run gf-download --dest data/ --dataset flickr2k
    uv run gf-download --dest data/ --dataset both
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

FLICKR2K_URL = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"


def progress_hook(count, block_size, total_size):
    pct = min(count * block_size * 100 / total_size, 100)
    mb  = count * block_size / 1e6
    print(f"\r  {pct:5.1f}%  ({mb:.1f} MB)", end="", flush=True)


def download_file(url: str, dest: Path):
    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()


def extract_zip(path: Path, dest: Path):
    print(f"  Extracting {path.name} ...")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)


def extract_tar(path: Path, dest: Path):
    import tarfile
    print(f"  Extracting {path.name} ...")
    with tarfile.open(path, "r") as t:
        t.extractall(dest)


def download_div2k(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    div2k = dest / "DIV2K"
    div2k.mkdir(exist_ok=True)

    for split, url in DIV2K_URLS.items():
        zip_path = dest / f"DIV2K_{split}_HR.zip"
        out_dir  = div2k / f"DIV2K_{split}_HR"

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [skip] {out_dir} already exists")
            continue

        download_file(url, zip_path)
        extract_zip(zip_path, div2k)
        zip_path.unlink()
        print(f"  ✓ DIV2K {split} → {out_dir}")

    n_train = len(list((div2k / "DIV2K_train_HR").glob("*.png")))
    n_valid = len(list((div2k / "DIV2K_valid_HR").glob("*.png")))
    print(f"\n  DIV2K: {n_train} train images, {n_valid} valid images")


def download_flickr2k(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "Flickr2K.tar"
    out_dir  = dest / "Flickr2K" / "Flickr2K_HR"

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [skip] {out_dir} already exists")
        return

    download_file(FLICKR2K_URL, tar_path)
    extract_tar(tar_path, dest)
    tar_path.unlink()

    n = len(list(out_dir.glob("*.png")))
    print(f"\n  Flickr2K: {n} images → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--dest",    default="./data", help="Where to save datasets")
    parser.add_argument("--dataset", default="div2k",
                        choices=["div2k", "flickr2k", "both"],
                        help="Which dataset to download")
    args = parser.parse_args()

    dest = Path(args.dest)
    print(f"\n{'='*55}")
    print(f"  Guided Fusion — Dataset Downloader")
    print(f"{'='*55}\n")

    if args.dataset in ("div2k", "both"):
        print("► DIV2K (800 train + 100 valid HQ images, ~7 GB)")
        download_div2k(dest)

    if args.dataset in ("flickr2k", "both"):
        print("\n► Flickr2K (2650 HQ images, ~30 GB — optional)")
        download_flickr2k(dest)

    print(f"\n{'='*55}")
    print("  Download complete!")
    print(f"\n  Now train with:")
    if args.dataset == "div2k":
        print("  uv run gf-train \\")
        print(f"    --train_dirs {dest}/DIV2K/DIV2K_train_HR \\")
        print(f"    --val_dirs   {dest}/DIV2K/DIV2K_valid_HR")
    elif args.dataset == "both":
        print("  uv run gf-train \\")
        print(f"    --train_dirs {dest}/DIV2K/DIV2K_train_HR {dest}/Flickr2K/Flickr2K_HR \\")
        print(f"    --val_dirs   {dest}/DIV2K/DIV2K_valid_HR")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
