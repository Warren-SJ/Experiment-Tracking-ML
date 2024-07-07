from pathlib import Path
import requests
import zipfile
import os

def acquire_data(image_path: Path,
                 url: str,
                 zip_name: str):
    """Download and extract a zip file from a URL to a specified directory.
    Args:
    image_path: Path to the directory where the data will be stored.
    url: URL of the zip file to download.
    zip_name: Name of the zip file to download.
    Returns:
    None"""
    data_dir = Path("data")
    image_path = data_dir / image_path
    if image_path.is_dir():
        print("Image directory already exists")
    else:
        image_path.mkdir(parents=True, exist_ok=True)
        print("Image directory created")
        with open (image_path / zip_name, "wb") as f:
            print("Downloading zip file")
            request = requests.get(url)
            f.write(request.content)
        with zipfile.ZipFile(image_path / zip_name, "r") as zip_ref:
            print("Extracting zip file")
            zip_ref.extractall(image_path)
    if os.path.exists(image_path / zip_name):       
        os.remove(image_path / zip_name)
    print("Done!")