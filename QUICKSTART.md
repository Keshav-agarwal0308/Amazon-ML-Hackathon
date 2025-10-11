# Quick Start Guide for Amazon ML Hackathon

print("🚀 Amazon ML Hackathon - Quick Start Guide")
print("=" * 50)

print("\n📁 Project Structure:")
print("✅ All folders created")
print("✅ All source files implemented")
print("✅ Training scripts ready")
print("✅ Validation scripts ready")

print("\n🔧 Setup Instructions:")
print("1. Install dependencies:")
print("   pip install -r requirements.txt")
print()
print("2. Place your dataset files in dataset/ folder:")
print("   - dataset/train.csv")
print("   - dataset/test.csv")
print()
print("3. Choose your training approach:")

print("\n📝 Option A: Text-Only Baseline (Fast - 5 minutes)")
print("   python scripts/train_text_baseline.py")
print("   → Creates: dataset/test_out_text_only.csv")

print("\n🖼️ Option B: Complete Pipeline (Text + Images - 30+ minutes)")
print("   python scripts/download_images.py")
print("   python scripts/train_complete.py")
print("   → Creates: dataset/test_out_blended.csv")

print("\n🔍 Validation:")
print("   python scripts/validate_submission.py")

print("\n📊 Data Exploration:")
print("   python notebooks/01_eda.py")

print("\n🎯 Key Features:")
print("✅ SMAPE metric implementation")
print("✅ GroupKFold for near-duplicate handling")
print("✅ Advanced text feature engineering (TF-IDF, IPQ, brand extraction)")
print("✅ ResNet50 image embeddings")
print("✅ Multiple models (Ridge, ElasticNet, LightGBM)")
print("✅ Intelligent model blending")
print("✅ Comprehensive validation")

print("\n🏆 Competition Strategy:")
print("Day 1: Submit text-only baseline")
print("Day 2: Add image features and blend")
print("Day 3: Feature engineering improvements")
print("Day 4: Final optimization and ensemble")

print("\n⚠️ Important Notes:")
print("- No external APIs or price lookups")
print("- All models use local libraries only")
print("- Well under 8B parameter limit")
print("- Reproducible with fixed random seeds")

print("\n🎉 Ready to win the hackathon!")
print("Start with: python scripts/train_text_baseline.py")
