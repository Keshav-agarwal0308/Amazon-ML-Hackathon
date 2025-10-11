import pandas as pd
from .blend import apply_blend

def main():
    _ = apply_blend(
        test_df=pd.read_csv('dataset/test.csv'),
        text_preds_parquet='features/test_text_preds.parquet',
        image_preds_parquet='features/test_image_preds.parquet',
        out_csv='dataset/test_out.csv'
    )
    print("Saved: dataset/test_out.csv")

if __name__ == '__main__':
    main()
