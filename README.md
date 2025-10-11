# Amazon ML Hackathon - Complete Solution

This repository contains a complete multimodal solution for the Amazon ML Hackathon that combines text and image features to predict product prices.

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your dataset files in the `dataset/` folder:
- `train.csv` - Training data with columns: sample_id, catalog_content, image_link, price
- `test.csv` - Test data with columns: sample_id, catalog_content, image_link

### 3. Run Complete Pipeline

#### Option A: Full Pipeline (Text + Images)
```bash
# Download images
python scripts/download_images.py

# Extract image features
python -c "
import pandas as pd
from src.fe_image import save_image_features
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
save_image_features(train, test, out_prefix='features/img')
"

# Train text models
python -c "
import pandas as pd
from src.model_text import train_text_models
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
preds_text = train_text_models(train, test, folds=None, outdir='models')
pd.DataFrame({'sample_id': test['sample_id'], **{k: v for k,v in preds_text.items()}}).to_parquet('features/test_text_preds.parquet', index=False)
"

# Train image model
python -c "
import pandas as pd
from src.model_image import train_image_model
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
preds_img = train_image_model(train, test, feat_prefix='features/img', out_path='models/img_lgbm.joblib')
pd.DataFrame({'sample_id': test['sample_id'], 'Image_LGBM': preds_img}).to_parquet('features/test_image_preds.parquet', index=False)
"

# Generate final submission
python -m src.infer_test
```

#### Option B: Text-Only Baseline (Fast)
```bash
# Train text models only
python -c "
import pandas as pd
from src.model_text import train_text_models
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
preds_text = train_text_models(train, test, folds=None, outdir='models')
# Create text-only submission
t = pd.DataFrame({'sample_id': test['sample_id'], **{k: v for k,v in preds_text.items()}})
out = t[['sample_id','Text_Ridge']].rename(columns={'Text_Ridge':'price'})
out.to_csv('dataset/test_out.csv', index=False)
"
```

## 📁 Project Structure

```
/dataset                 # train.csv, test.csv, sample_test_out.csv
/src
  utils.py               # Image downloading utilities
  smape.py              # SMAPE metric implementation
  folds.py              # GroupKFold for near-duplicates
  fe_text.py            # Text feature engineering
  fe_image.py           # Image feature extraction (ResNet50)
  model_text.py         # Text models (Ridge/ElasticNet/LightGBM)
  model_image.py        # Image regression model
  blend.py              # Model blending
  infer_test.py         # Final inference script
/assets/images          # Downloaded images
/features               # Cached features (parquet/npy)
/models                 # Trained model artifacts
/notebooks              # EDA and experimentation notebooks
/scripts                # Utility scripts
```

## 🧠 Model Architecture

### Text Features
- **TF-IDF**: Word and character n-grams
- **Numeric Features**: Text length, digit ratio, IPQ (items per quantity)
- **Brand Extraction**: Regex-based brand identification
- **Unit Parsing**: Volume, weight, dimension extraction
- **Flags**: Bundle, refill, mini, premium indicators

### Image Features
- **ResNet50**: Pre-trained CNN for feature extraction
- **PCA**: Dimensionality reduction to 256 components
- **Missing Flag**: Binary indicator for missing images

### Models
1. **Ridge Regression**: Linear baseline with L2 regularization
2. **ElasticNet**: Linear model with L1+L2 regularization
3. **LightGBM**: Gradient boosting on text features
4. **Image LightGBM**: Gradient boosting on image embeddings

### Blending
- Weighted ensemble of all models
- Grid search for optimal weights
- Ensures positive predictions

## 📊 Performance Tips

### Day 1: Text Baseline
1. Start with text-only models
2. Submit Ridge regression as baseline
3. Begin image downloading in background

### Day 2: Add Images
1. Complete image feature extraction
2. Train image model
3. Create first blended submission

### Day 3: Feature Engineering
1. Add more text flags (XL, organic, value pack, etc.)
2. Experiment with different TF-IDF parameters
3. Try different blending weights

### Day 4: Final Optimization
1. Cross-validation for weight selection
2. Ensemble multiple configurations
3. Final submission with best weights

## 🔧 Customization

### Adding New Text Features
Edit `src/fe_text.py`:
```python
def basic_flags(text: str):
    t = str(text).lower()
    flags = {
        'is_bundle': int(any(w in t for w in ['bundle','combo','set of','pack of','x'])),
        'is_refill': int('refill' in t or 'refills' in t),
        'is_mini': int(any(w in t for w in ['mini','travel size','trial'])),
        'is_premium': int(any(w in t for w in ['premium','luxury','pro','ultra'])),
        # Add your custom flags here
        'is_organic': int('organic' in t),
        'is_xl': int('xl' in t or 'extra large' in t),
    }
    return flags
```

### Adjusting Model Parameters
Edit `src/model_text.py`:
```python
# Ridge parameters
ridge = Ridge(alpha=2.0, random_state=42)  # Try 1.0, 3.0, 5.0

# ElasticNet parameters  
enet = ElasticNet(alpha=0.01, l1_ratio=0.15, random_state=42, max_iter=2000)

# LightGBM parameters
params = dict(
    objective='mae',
    learning_rate=0.07,  # Try 0.05, 0.1
    num_leaves=128,      # Try 64, 256
    # ... other params
)
```

## 📈 Monitoring

### Check Submission Format
```python
import pandas as pd
te = pd.read_csv('dataset/test.csv')
out = pd.read_csv('dataset/test_out.csv')
assert set(out.columns)=={'sample_id','price'}
assert len(out)==len(te)
assert out['price'].min()>0
assert out['sample_id'].is_unique
print("✅ Submission format is correct!")
```

### Track Performance
Keep a log of submissions:
- Submission name
- Model parameters
- CV score (if available)
- Notes

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce batch size in image processing
   ```python
   embed_images(df, batch_size=16)  # Instead of 32
   ```

3. **Missing Images**: Check download logs and retry failed downloads

4. **Slow Training**: Use fewer features or smaller models for initial testing

### Performance Optimization

1. **Parallel Processing**: Set `n_jobs=-1` in LightGBM
2. **Feature Caching**: Save intermediate features to disk
3. **Model Caching**: Save trained models to avoid retraining

## 🏆 Competition Strategy

### Submission Schedule
- **Day 1**: Text-only baseline (Ridge)
- **Day 2**: Text ensemble (Ridge + ElasticNet + LightGBM)
- **Day 3**: Multimodal blend (Text + Images)
- **Day 4**: Optimized ensemble with best weights

### Risk Management
- Always keep a text-only fallback
- Submit multiple variants
- Monitor leaderboard trends
- Don't overfit to public leaderboard

## 📝 Notes

- **No External APIs**: All models use local libraries only
- **No Price Lookups**: Only uses provided dataset features
- **Model Size**: Well under 8B parameter limit
- **Reproducible**: Fixed random seeds throughout

Good luck with the hackathon! 🎉