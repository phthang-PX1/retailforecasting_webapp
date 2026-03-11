import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def clean_raw_data(sales_path, skus_path):
    """Đọc và làm sạch dữ liệu thô ban đầu."""
    print("[1/5] Đang đọc và làm sạch dữ liệu thô...")
    df_sales = pd.read_csv(sales_path)
    df_skus = pd.read_csv(skus_path)

    df_sales['date'] = pd.to_datetime(df_sales['date'])
    df_sales = df_sales.drop_duplicates()
    
    if 'customer_id' in df_sales.columns:
        df_sales['customer_id'] = df_sales['customer_id'].fillna(0).astype(int)
        
    return df_sales, df_skus