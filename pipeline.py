import sys
import os

# Thêm thư mục gốc vào đường dẫn hệ thống để Python tìm thấy thư mục 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_cleaning import clean_raw_data
from src.data.aggregation import aggregate_to_weekly
from src.features.feature_engineering import create_features
from src.models.train_model import train_and_save_lightgbm

def run_all():
    print("="*60)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH")
    print("="*60)
    
    # Đường dẫn file
    sales_raw_path = 'data/raw/bm_sales_synthetic.csv'
    skus_raw_path = 'data/raw/bm_skus.csv'
    features_out_path = 'data/features/model_features.csv'
    model_out_path = 'models/lightgbm_discount_model.pkl'
    
    # Đảm bảo các thư mục tồn tại
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Làm sạch
    df_sales, df_skus = clean_raw_data(sales_raw_path, skus_raw_path)
    
    # 2. Tổng hợp
    weekly_df = aggregate_to_weekly(df_sales)
    weekly_df.to_csv('data/processed/weekly_clean.csv', index=False)
    
    # 3. Tạo đặc trưng
    features_df = create_features(weekly_df, df_skus)
    features_df.to_csv(features_out_path, index=False)
    
    # 4. Huấn luyện Model
    train_and_save_lightgbm(features_df, model_out_path)
    
    print("="*60)
    print("✅ PIPELINE HOÀN TẤT! Backend đã sẵn sàng phục vụ Web App.")
    print("="*60)

if __name__ == "__main__":
    run_all()