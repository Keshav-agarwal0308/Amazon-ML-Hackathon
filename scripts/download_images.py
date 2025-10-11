import os
import pandas as pd
from src.utils import download_images  # provided by organizers

os.makedirs('assets/images', exist_ok=True)

df_tr = pd.read_csv('dataset/train.csv')
df_te = pd.read_csv('dataset/test.csv')
urls = pd.concat([df_tr[['sample_id','image_link']], df_te[['sample_id','image_link']]])
download_images(urls, out_dir='assets/images', retries=3, sleep_seconds=0.2)
