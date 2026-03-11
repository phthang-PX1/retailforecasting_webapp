import lightgbm as lgb
import joblib

def train_and_save_lightgbm(df_features, model_save_path):
    """Huấn luyện mô hình LightGBM với các siêu tham số đã tối ưu và ràng buộc đơn điệu."""
    print("[4/5] Đang huấn luyện mô hình Machine Learning (LightGBM)...")
    
    features = [
        "store_id", "sku_id", "discount_pct", "weekofyear", "month",
        "quarter", "lag_1", "lag_2", "lag_4", "rolling_mean_2",
        "rolling_mean_4", "store_weekly_sales", "sku_weekly_sales"
    ]
    target = "log_quantity"
    
    # Ràng buộc đơn điệu: Tăng discount (vị trí 2) -> Tăng Quantity
    constraints = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.072, num_leaves=98,
        subsample=0.624, colsample_bytree=0.887, min_child_samples=72,        
        random_state=42, monotone_constraints=constraints,
        monotone_constraints_method='advanced', n_jobs=-1
    )
    
    model.fit(df_features[features], df_features[target])
    
    joblib.dump(model, model_save_path)
    print(f"[5/5] Đã lưu mô hình thành công tại: {model_save_path}")
    
    return model