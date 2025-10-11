# Example EDA Notebook for Amazon ML Hackathon

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.folds import make_group_folds
from src.fe_text import extract_ipq, extract_brand, parse_units, basic_flags

# Load data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print("Dataset shapes:")
print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

# Basic statistics
print("\nPrice statistics:")
print(train['price'].describe())

# Text analysis
print("\nText length analysis:")
train['text_length'] = train['catalog_content'].astype(str).apply(len)
print(train['text_length'].describe())

# Feature engineering examples
print("\nFeature engineering examples:")
train['ipq'] = train['catalog_content'].apply(extract_ipq)
train['brand'] = train['catalog_content'].apply(extract_brand)
train['units'] = train['catalog_content'].apply(parse_units)
train['flags'] = train['catalog_content'].apply(basic_flags)

print("IPQ distribution:")
print(train['ipq'].value_counts().head(10))

print("\nTop brands:")
print(train['brand'].value_counts().head(10))

# Create folds
print("\nCreating folds...")
train['fold'] = make_group_folds(train, n_splits=5)
print("Fold distribution:")
print(train['fold'].value_counts().sort_index())

# Save folds for later use
train[['sample_id','fold']].to_csv('features/folds.csv', index=False)
print("Saved folds to features/folds.csv")

# Price distribution by fold
plt.figure(figsize=(12, 8))
for fold in range(5):
    plt.subplot(2, 3, fold+1)
    train[train['fold']==fold]['price'].hist(bins=50, alpha=0.7)
    plt.title(f'Fold {fold} Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('notebooks/price_distribution_by_fold.png', dpi=150, bbox_inches='tight')
plt.show()

print("EDA complete! Check notebooks/price_distribution_by_fold.png")
