import os 
import zipfile 

from pathlib import Path

import requests

# Create directory if needed
data_path = Path(".data")
if not data_path.is_dir():
    print(f"Creating the '{str(data_path)}' directory...")
    data_path.mkdir(parents=True)
else:
    print(f"Directory '{str(data_path)}' already exists, skipping this step.")

# Download the data if needed 
if not any(data_path.iterdir()): 
    with open(data_path/"pizza_steak_sushi.zip", "wb") as f:
        print(f"Downloading the zip data file...")
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)

    # Unzipping data
    image_path = data_path/"pizza_steak_sushi"
    with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip") as zip_f:
        print("Unzipping data...")
        zip_f.extractall(image_path)
    print(f"Removing zipfile...")
    os.remove(data_path/"pizza_steak_sushi.zip")
else:
    print(f"Data already downloaded, skipping this step.")
