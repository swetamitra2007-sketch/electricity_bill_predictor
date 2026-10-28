"""
=============================================================================
STAGE 1: Official Dataset Downloader
=============================================================================
This script automatically downloads the official "Individual Household Electric 
Power Consumption" dataset from the UCI Machine Learning Repository and extracts 
the raw data file into the 'data/' folder.

Dataset Source:
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
=============================================================================
"""

import os
import sys
import zipfile
import urllib.request
import shutil

# Target file paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "household_power_consumption.txt")
TEMP_ZIP_FILE = os.path.join(DATA_DIR, "household_power_consumption.zip")

# Primary and backup URLs for the official UCI dataset
DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
]


def download_progress_hook(block_num, block_size, total_size):
    """
    Displays a real-time progress bar in the terminal while downloading.
    """
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100.0)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading dataset: {mb_downloaded:.1f} MB / {mb_total:.1f} MB [{percent:.1f}%]")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloading dataset: {mb_downloaded:.1f} MB downloaded...")
    sys.stdout.flush()


def download_and_extract_dataset():
    """
    Checks if the dataset already exists. If not, downloads and extracts it.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_DATA_FILE) and os.path.getsize(RAW_DATA_FILE) > 10 * 1024 * 1024:
        size_mb = os.path.getsize(RAW_DATA_FILE) / (1024 * 1024)
        print(f"[OK] Raw dataset already exists at: {RAW_DATA_FILE} ({size_mb:.1f} MB)")
        return RAW_DATA_FILE

    print("Official dataset not found locally. Starting automated download from UCI...")

    download_success = False
    for url in DATASET_URLS:
        try:
            print(f"Connecting to: {url}")
            # Use custom User-Agent to avoid HTTP 403 blocks
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as response, open(TEMP_ZIP_FILE, 'wb') as out_file:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 64  # 64 KB chunks
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                    else:
                        sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f} MB downloaded...")
                    sys.stdout.flush()
                print()
            download_success = True
            break
        except Exception as e:
            print(f"\n[Warning] Could not download from {url}: {e}")
            if os.path.exists(TEMP_ZIP_FILE):
                os.remove(TEMP_ZIP_FILE)

    if not download_success:
        print("\n" + "=" * 70)
        print("ERROR: Automatic download failed.")
        print("Please download the dataset manually using this link:")
        print("https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip")
        print(f"Extract 'household_power_consumption.txt' into: {DATA_DIR}")
        print("=" * 70)
        sys.exit(1)

    print(f"Download complete. Extracting '{TEMP_ZIP_FILE}'...")
    try:
        with zipfile.ZipFile(TEMP_ZIP_FILE, 'r') as zip_ref:
            # Find the text file inside the zip archive
            txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
            if not txt_files:
                raise FileNotFoundError("No .txt file found inside the downloaded zip archive.")
            target_inner_file = txt_files[0]
            zip_ref.extract(target_inner_file, DATA_DIR)
            
            extracted_path = os.path.join(DATA_DIR, target_inner_file)
            if extracted_path != RAW_DATA_FILE:
                shutil.move(extracted_path, RAW_DATA_FILE)

        # Remove the zip file after successful extraction to save space
        if os.path.exists(TEMP_ZIP_FILE):
            os.remove(TEMP_ZIP_FILE)

        size_mb = os.path.getsize(RAW_DATA_FILE) / (1024 * 1024)
        print(f"[SUCCESS] Dataset extracted to: {RAW_DATA_FILE} ({size_mb:.1f} MB)")
        return RAW_DATA_FILE

    except Exception as e:
        print(f"[ERROR] Failed to extract archive: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_and_extract_dataset()
