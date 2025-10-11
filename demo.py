# Complete Amazon ML Hackathon Solution Demo

"""
This script demonstrates the complete workflow for the Amazon ML Hackathon.
Run this after placing your dataset files in the dataset/ folder.
"""

import os
import sys
import pandas as pd
import numpy as np

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'lightgbm', 
        'pyarrow', 'torch', 'torchvision', 'Pillow', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_dataset():
    """Check if dataset files exist"""
    required_files = ['dataset/train.csv', 'dataset/test.csv']
    missing = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing dataset files: {missing}")
        print("Please place your dataset files in the dataset/ folder")
        return False
    
    print("✅ Dataset files found")
    
    # Quick dataset info
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    print(f"📊 Train: {train.shape}, Test: {test.shape}")
    print(f"💰 Price range: {train['price'].min():.2f} - {train['price'].max():.2f}")
    
    return True

def run_text_baseline():
    """Run text-only baseline training"""
    print("\n📝 Running text-only baseline...")
    
    try:
        # Import and run text training
        from src.model_text import train_text_models
        
        train = pd.read_csv('dataset/train.csv')
        test = pd.read_csv('dataset/test.csv')
        
        print("Training text models...")
        preds_text = train_text_models(train, test, folds=None, outdir='models')
        
        # Save predictions
        pred_df = pd.DataFrame({
            'sample_id': test['sample_id'],
            **{k: v for k,v in preds_text.items()}
        })
        pred_df.to_parquet('features/test_text_preds.parquet', index=False)
        
        # Create submission
        submission = pd.DataFrame({
            'sample_id': test['sample_id'],
            'price': preds_text['Text_Ridge']
        })
        submission.to_csv('dataset/test_out_text_only.csv', index=False)
        
        print("✅ Text baseline complete!")
        print(f"📁 Saved: dataset/test_out_text_only.csv")
        print(f"💰 Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text training: {e}")
        return False

def main():
    print("🚀 Amazon ML Hackathon - Complete Solution Demo")
    print("=" * 60)
    
    # Step 1: Check requirements
    print("\n1️⃣ Checking requirements...")
    if not check_requirements():
        return
    
    # Step 2: Check dataset
    print("\n2️⃣ Checking dataset...")
    if not check_dataset():
        return
    
    # Step 3: Run text baseline
    print("\n3️⃣ Running text baseline...")
    if not run_text_baseline():
        return
    
    # Step 4: Validation
    print("\n4️⃣ Validating submission...")
    try:
        from scripts.validate_submission import validate_submission
        if validate_submission('dataset/test_out_text_only.csv'):
            print("✅ Submission is valid and ready!")
        else:
            print("❌ Submission validation failed")
    except Exception as e:
        print(f"⚠️ Could not validate: {e}")
    
    # Final instructions
    print("\n🎉 Demo Complete!")
    print("\n📋 Next Steps:")
    print("1. Submit dataset/test_out_text_only.csv for text-only baseline")
    print("2. For multimodal solution, run:")
    print("   python scripts/download_images.py")
    print("   python scripts/train_complete.py")
    print("3. Explore data with: python notebooks/01_eda.py")
    print("4. Validate submissions with: python scripts/validate_submission.py")
    
    print("\n🏆 You're ready to compete!")
    print("This solution includes:")
    print("✅ Advanced text feature engineering")
    print("✅ Multiple ML models (Ridge, ElasticNet, LightGBM)")
    print("✅ Image processing with ResNet50")
    print("✅ Intelligent model blending")
    print("✅ Comprehensive validation")
    print("✅ No external APIs (rule compliant)")

if __name__ == '__main__':
    main()
