import os
import io
import time
import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm

def download_images(urls_df, out_dir='assets/images', retries=3, sleep_seconds=0.2):
    """
    Download images from URLs and save them locally.
    
    Args:
        urls_df: DataFrame with columns 'sample_id' and 'image_link'
        out_dir: Directory to save images
        retries: Number of retry attempts for failed downloads
        sleep_seconds: Sleep time between requests
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, row in tqdm(urls_df.iterrows(), total=len(urls_df), desc="Downloading images"):
        sample_id = row['sample_id']
        image_url = row['image_link']
        
        if pd.isna(image_url) or not image_url:
            continue
            
        output_path = os.path.join(out_dir, f"{sample_id}.jpg")
        
        # Skip if already downloaded
        if os.path.exists(output_path):
            continue
            
        for attempt in range(retries):
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                # Try to open as image to validate
                img = Image.open(io.BytesIO(response.content))
                img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=85)
                break
                
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to download {sample_id}: {e}")
                else:
                    time.sleep(sleep_seconds)
        
        time.sleep(sleep_seconds)
