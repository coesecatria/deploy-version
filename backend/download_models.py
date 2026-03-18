import os
import requests
import zipfile
from tqdm import tqdm

# Configuration
MODELS_DIR = os.path.join(os.getcwd(), "models", "insightface")
ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_PATH = os.path.join(MODELS_DIR, "buffalo_l.zip")

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    print("Download complete.")

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    if not os.path.exists(ZIP_PATH):
        download_file(ZIP_URL, ZIP_PATH)
    else:
        print("Zip file already exists, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODELS_DIR)
    
    print(f"Extracted to {MODELS_DIR}")
    
    # List files to verify
    print("Files in directory:")
    for f in os.listdir(MODELS_DIR):
        print(f" - {f}")

if __name__ == "__main__":
    main()
