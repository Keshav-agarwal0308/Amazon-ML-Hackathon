# Submission Validation Script

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def validate_submission(submission_path):
    """Validate submission format and content"""
    print(f"🔍 Validating {submission_path}...")
    
    if not os.path.exists(submission_path):
        print(f"❌ File not found: {submission_path}")
        return False
    
    try:
        # Load submission
        sub = pd.read_csv(submission_path)
        
        # Check basic format
        print(f"📊 Shape: {sub.shape}")
        print(f"📋 Columns: {list(sub.columns)}")
        
        # Required columns
        if set(sub.columns) != {'sample_id', 'price'}:
            print(f"❌ Wrong columns. Expected: ['sample_id', 'price'], Got: {list(sub.columns)}")
            return False
        
        # Check sample_id uniqueness
        if not sub['sample_id'].is_unique:
            print("❌ sample_id column contains duplicates")
            return False
        
        # Check for missing values
        if sub.isnull().sum().sum() > 0:
            print("❌ Submission contains missing values")
            print(sub.isnull().sum())
            return False
        
        # Check price positivity
        if sub['price'].min() <= 0:
            print(f"❌ Prices must be positive. Min price: {sub['price'].min()}")
            return False
        
        # Check price range
        print(f"💰 Price range: {sub['price'].min():.2f} - {sub['price'].max():.2f}")
        print(f"📈 Price mean: {sub['price'].mean():.2f}")
        print(f"📊 Price std: {sub['price'].std():.2f}")
        
        # Compare with test set
        if os.path.exists('dataset/test.csv'):
            test = pd.read_csv('dataset/test.csv')
            if len(sub) != len(test):
                print(f"❌ Length mismatch. Test: {len(test)}, Submission: {len(sub)}")
                return False
            
            missing_ids = set(test['sample_id']) - set(sub['sample_id'])
            if missing_ids:
                print(f"❌ Missing sample_ids: {len(missing_ids)}")
                return False
            
            extra_ids = set(sub['sample_id']) - set(test['sample_id'])
            if extra_ids:
                print(f"❌ Extra sample_ids: {len(extra_ids)}")
                return False
        
        print("✅ Submission validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading submission: {e}")
        return False

def main():
    print("🚀 Amazon ML Hackathon - Submission Validator")
    
    # Check if test data exists
    if not os.path.exists('dataset/test.csv'):
        print("❌ dataset/test.csv not found!")
        return
    
    test = pd.read_csv('dataset/test.csv')
    print(f"📊 Test set has {len(test)} samples")
    
    # Validate all submission files
    submission_files = [
        'dataset/test_out_text_only.csv',
        'dataset/test_out_blended.csv',
        'dataset/test_out.csv'
    ]
    
    valid_submissions = []
    for file_path in submission_files:
        if os.path.exists(file_path):
            if validate_submission(file_path):
                valid_submissions.append(file_path)
            print("-" * 50)
    
    print(f"\n📋 Summary:")
    print(f"Valid submissions: {len(valid_submissions)}")
    for sub in valid_submissions:
        print(f"  ✅ {sub}")
    
    if valid_submissions:
        print(f"\n🎉 Ready to submit! Use: {valid_submissions[0]}")
    else:
        print("\n⚠️ No valid submissions found. Run training pipeline first.")

if __name__ == '__main__':
    main()
