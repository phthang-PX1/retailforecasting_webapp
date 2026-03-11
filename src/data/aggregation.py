import pandas as pd

def aggregate_to_weekly(df_sales):
    """Gom nhóm dữ liệu bán hàng theo tuần để giảm nhiễu."""
    print("[2/5] Đang tổng hợp dữ liệu theo tuần (Weekly Aggregation)...")
    
    df_sales['week'] = df_sales['date'].dt.to_period("W").apply(lambda r: r.start_time)
    
    weekly = df_sales.groupby(["week", "store_id", "sku_id"], as_index=False).agg({
        "quantity": "sum",
        "total_value": "sum",
        "discount_pct": "mean" 
    }).sort_values(["store_id", "sku_id", "week"]).reset_index(drop=True)

    # Lọc các cặp Store-SKU có ít nhất 20 tuần dữ liệu để đảm bảo mô hình học được
    pair_count = weekly.groupby(["store_id", "sku_id"])["quantity"].count().reset_index(name="n_obs")
    valid_pairs = pair_count[pair_count["n_obs"] >= 20]
    weekly = weekly.merge(valid_pairs[["store_id", "sku_id"]], on=["store_id", "sku_id"], how="inner")

    return weekly