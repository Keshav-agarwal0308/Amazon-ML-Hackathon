# Complete Training Pipeline Script

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.model_text import train_text_models
from src.model_image import train_image_model
from src.fe_image import save_image_features
from src.blend import apply_blend
from src.smape import smape

def main():
    print("🚀 Starting Amazon ML Hackathon Training Pipeline")
    
    # Check if dataset exists
    if not os.path.exists('dataset/train.csv'):
        print("❌ Error: dataset/train.csv not found!")
        print("Please place your dataset files in the dataset/ folder")
        return
    
    print("📊 Loading data...")
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Create output directories
    os.makedirs('features', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Train text models
    print("\n📝 Training text models...")
    preds_text = train_text_models(train, test, folds=None, outdir='models')
    
    # Save text predictions
    pred_df = pd.DataFrame({
        'sample_id': test['sample_id'],
        **{k: v for k,v in preds_text.items()}
    })
    pred_df.to_parquet('features/test_text_preds.parquet', index=False)
    print("✅ Text predictions saved to features/test_text_preds.parquet")
    
    # Create text-only submission
    text_submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': preds_text['Text_Ridge']
    })
    text_submission.to_csv('dataset/test_out_text_only.csv', index=False)
    print("✅ Text-only submission saved to dataset/test_out_text_only.csv")
    
    # Step 2: Check if images are available
    if os.path.exists('assets/images') and len(os.listdir('assets/images')) > 0:
        print("\n🖼️ Processing images...")
        
        # Extract image features
        save_image_features(train, test, out_prefix='features/img')
        print("✅ Image features extracted")
        
        # Train image model
        preds_img = train_image_model(train, test, feat_prefix='features/img', out_path='models/img_lgbm.joblib')
        
        # Save image predictions
        img_pred_df = pd.DataFrame({
            'sample_id': test['sample_id'],
            'Image_LGBM': preds_img
        })
        img_pred_df.to_parquet('features/test_image_preds.parquet', index=False)
        print("✅ Image predictions saved to features/test_image_preds.parquet")
        
        # Step 3: Create blended submission
        print("\n🔄 Creating blended submission...")
        apply_blend(
            test_df=None,
            text_preds_parquet='features/test_text_preds.parquet',
            image_preds_parquet='features/test_image_preds.parquet',
            out_csv='dataset/test_out_blended.csv'
        )
        print("✅ Blended submission saved to dataset/test_out_blended.csv")
        
    else:
        print("\n⚠️ No images found in assets/images/")
        print("Run 'python scripts/download_images.py' first to download images")
    
    # Validation
    print("\n🔍 Validating submissions...")
    
    # Check text-only submission
    text_sub = pd.read_csv('dataset/test_out_text_only.csv')
    print(f"Text-only submission: {text_sub.shape}, price range: {text_sub['price'].min():.2f}-{text_sub['price'].max():.2f}")
    
    # Check blended submission if it exists
    if os.path.exists('dataset/test_out_blended.csv'):
        blend_sub = pd.read_csv('dataset/test_out_blended.csv')
        print(f"Blended submission: {blend_sub.shape}, price range: {blend_sub['price'].min():.2f}-{blend_sub['price'].max():.2f}")
    
    print("\n🎉 Training pipeline complete!")
    print("\nNext steps:")
    print("1. Submit dataset/test_out_text_only.csv for text-only baseline")
    if os.path.exists('dataset/test_out_blended.csv'):
        print("2. Submit dataset/test_out_blended.csv for multimodal solution")
    print("3. Experiment with different parameters in the source files")
    print("4. Check notebooks/01_eda.py for data exploration")

if __name__ == '__main__':
    main()
