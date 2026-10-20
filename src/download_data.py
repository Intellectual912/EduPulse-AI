#!/usr/bin/env python3
"""
EduPulse AI — dataset acquisition helper (Milestone 1).

Downloads the Open University Learning Analytics Dataset (OULAD), verifies its
MD5 checksum against the value published with the dataset, extracts the seven
CSVs into `data/raw/` and never overwrites an existing raw file.

The dataset is CC BY 4.0. Cite:
    Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning
    Analytics dataset. Scientific Data, 4, 170171.
    https://doi.org/10.1038/sdata.2017.171

Usage:
    python src/download_data.py               # download + verify, keep the archive
    python src/download_data.py --extract     # also unpack the seven CSVs
    python src/download_data.py --repack      # delete unpacked CSVs, keep the archive
    python src/download_data.py --check-only  # report what is available

By default the verified archive is kept and NOT unpacked: `src/preprocess.py`
reads the seven tables straight out of `anonymisedData.zip`, so the ~444 MB of
extracted CSV (453 MB of which is studentVle.csv) never has to exist on disk.
That keeps the project inside tight workspace quotas. Use `--extract` when you
genuinely need the plain files (e.g. to inspect them in another tool), and
`--repack` to reclaim the space afterwards.

If your environment has no network access, download the archive manually and
drop `anonymisedData.zip` into data/raw/, then run this script again: it will
verify the checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_NAME = "anonymisedData.zip"

# MD5 published with the dataset; independently re-verified on 2026-10-20.
EXPECTED_MD5 = "7412686fd77cf0e0ee1e8c3e9b354308"

# Mirrors are tried in order. The official page's download link has moved
# several times, so the authors' Figshare record and the UCI repository are
# used as durable fallbacks. All three serve the same archive.
MIRRORS = [
    ("OU Analyse (official page)",
     "https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip"),
    ("Figshare (authors' record, DOI 10.6084/m9.figshare.5081998)",
     "https://ndownloader.figshare.com/files/8606371"),
    ("UCI ML Repository (dataset 349)",
     "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"),
]

EXPECTED_FILES = [
    "studentInfo.csv", "studentAssessment.csv", "assessments.csv", "studentVle.csv",
    "studentRegistration.csv", "vle.csv", "courses.csv",
]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EduPulse-AI setup)"})
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
    print()


def extracted_files() -> list[Path]:
    return [RAW_DIR / f for f in EXPECTED_FILES if (RAW_DIR / f).exists()]


def archive_status() -> tuple[bool, str]:
    """Return (has_archive, md5-or-reason) for data/raw/anonymisedData.zip."""
    zp = RAW_DIR / ZIP_NAME
    if not zp.exists():
        return False, "not present"
    digest = md5_of(zp)
    if digest != EXPECTED_MD5:
        return False, f"checksum mismatch ({digest})"
    return True, digest


def check_existing() -> bool:
    """Report what is usable. The pipeline works with either layout."""
    zp_ok, detail = archive_status()
    files = extracted_files()

    if len(files) == len(EXPECTED_FILES):
        print(f"extracted CSVs present in {RAW_DIR.relative_to(PROJECT_ROOT)}:")
        for f in files:
            print(f"  {f.name:26s} {f.stat().st_size/1e6:8.2f} MB")
        if zp_ok:
            print(f"  ({ZIP_NAME} also present, MD5 verified)")
        return True

    if zp_ok:
        print(f"archive {ZIP_NAME} present and MD5-verified ({detail}).")
        print("src/preprocess.py reads the tables directly from it - no extraction needed.")
        if files:
            print(f"note: {len(files)} extracted CSV(s) also present (see --repack to reclaim space).")
        return True

    print(f"No usable dataset in data/raw (archive: {detail}; "
          f"extracted CSVs: {len(files)}/{len(EXPECTED_FILES)}).")
    return False


def extract_archive(zip_path: Path) -> None:
    print("Extracting into data/raw (existing CSVs are kept, never overwritten) ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name.endswith(".csv"):
                continue
            target = RAW_DIR / name
            if target.exists():
                print(f"  keep   {name} (already present)")
                continue
            with zf.open(member) as src, target.open("wb") as out:
                out.write(src.read())
            print(f"  wrote  {name}")


def repack() -> int:
    """Delete the extracted CSVs, keeping the verified archive."""
    files = extracted_files()
    if not files:
        print("Nothing to remove: no extracted CSVs in data/raw.")
        return 0
    if not archive_status()[0]:
        print("Refusing to delete the extracted CSVs: no verified archive to fall back on.\n"
              "Run `python src/download_data.py` first.", file=sys.stderr)
        return 2
    freed = 0
    for f in files:
        freed += f.stat().st_size
        f.unlink()
        print(f"  removed {f.name}")
    print(f"Freed {freed/1e6:.1f} MB. The pipeline now reads from {ZIP_NAME} directly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download, verify and (optionally) unpack the OULAD dataset.")
    ap.add_argument("--check-only", action="store_true",
                    help="report what is available, change nothing")
    ap.add_argument("--extract", action="store_true",
                    help="unpack the seven CSVs into data/raw (uses ~444 MB)")
    ap.add_argument("--repack", action="store_true",
                    help="delete unpacked CSVs, keeping the verified archive")
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.repack:
        return repack()

    if args.check_only:
        return 0 if check_existing() else 1

    zip_path = RAW_DIR / ZIP_NAME
    have_archive, detail = archive_status()

    if not have_archive:
        if zip_path.exists():
            print(f"Existing archive fails the checksum ({detail}); re-downloading.")
            zip_path.unlink()
        ok = False
        for label, url in MIRRORS:
            print(f"Trying {label} ...")
            try:
                download(url, zip_path)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                print(f"  failed: {exc}")
                zip_path.unlink(missing_ok=True)
                continue
            got = md5_of(zip_path)
            if got == EXPECTED_MD5:
                print(f"  MD5 verified: {got}")
                ok = True
                break
            print(f"  checksum mismatch: got {got}, expected {EXPECTED_MD5}; discarding")
            zip_path.unlink(missing_ok=True)
        if not ok:
            print("\nCould not obtain a verified archive.", file=sys.stderr)
            print("Download it manually from https://research.stem.open.ac.uk/ouanalyse/dataset/",
                  file=sys.stderr)
            print(f"and place it at {zip_path}, then re-run this script.", file=sys.stderr)
            return 2
    else:
        print(f"Archive already present and verified ({detail}).")

    if args.extract:
        extract_archive(zip_path)
    elif len(extracted_files()) == len(EXPECTED_FILES):
        print(f"\nExtracted CSVs already present; {ZIP_NAME} kept for reproducibility.")
    else:
        print("\nKeeping the archive packed: src/preprocess.py reads the tables directly "
              "from it.\nUse --extract if you need the plain CSVs on disk.")

    return 0 if check_existing() else 1


if __name__ == "__main__":
    raise SystemExit(main())
