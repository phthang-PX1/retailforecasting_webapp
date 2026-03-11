import pandas as pd
import numpy as np

def create_features(weekly_df, df_skus):
    """Tạo các biến Time-based, Lag, Rolling và chuẩn bị Target."""
    print("[3/5] Đang thực hiện Feature Engineering...")
    
    weekly_df["net_price"] = weekly_df["total_value"] / weekly_df["quantity"]
    weekly_df["weekofyear"] = weekly_df["week"].dt.isocalendar().week.astype(int)
    weekly_df["month"] = weekly_df["week"].dt.month
    weekly_df["quarter"] = weekly_df["week"].dt.quarter

    # Lag & Rolling Features
    for lag in [1, 2, 4]:
        weekly_df[f"lag_{lag}"] = weekly_df.groupby(["store_id", "sku_id"])["quantity"].shift(lag)
    for w in [2, 4]:
        weekly_df[f"rolling_mean_{w}"] = weekly_df.groupby(["store_id", "sku_id"])["quantity"].shift(1).rolling(w).mean()

    # Sức mua cấp độ Store và SKU
    store_weekly = weekly_df.groupby(["store_id", "week"], as_index=False)["quantity"].sum()
    store_weekly["store_weekly_sales"] = store_weekly.groupby("store_id")["quantity"].shift(1)
    
    sku_weekly = weekly_df.groupby(["sku_id", "week"], as_index=False)["quantity"].sum()
    sku_weekly["sku_weekly_sales"] = sku_weekly.groupby("sku_id")["quantity"].shift(1)

    weekly_df = weekly_df.merge(store_weekly[["store_id", "week", "store_weekly_sales"]], on=["store_id", "week"], how="left")
    weekly_df = weekly_df.merge(sku_weekly[["sku_id", "week", "sku_weekly_sales"]], on=["sku_id", "week"], how="left")

    # Xử lý Target (Log transform) và ghép Cost Price
    weekly_df["log_quantity"] = np.log1p(weekly_df["quantity"])
    weekly_df["store_weekly_sales"] = np.log1p(weekly_df["store_weekly_sales"])
    weekly_df["sku_weekly_sales"] = np.log1p(weekly_df["sku_weekly_sales"])
    
    weekly_df = weekly_df.merge(df_skus[["sku_id", "cost_price"]], on="sku_id", how="left")
    weekly_df['cost_price'] = weekly_df['cost_price'].fillna(0)
    
    # Loại bỏ giá trị Null sinh ra do dịch chuyển thời gian (shift)
    final_features_df = weekly_df.dropna().reset_index(drop=True)
    
    return final_features_df