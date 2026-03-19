import pandas as pd
import numpy as np
from src.models.predict import predict_sales

def simulate_what_if(store_id, sku_id, df_features, model, feature_cols, target_week="Next Week"):
    """Mô phỏng kịch bản What-If cho 1 cặp Store-SKU."""
    store_sku_data = df_features[(df_features['store_id'] == store_id) & 
                                 (df_features['sku_id'] == sku_id)].sort_values('week')
    if store_sku_data.empty: return None

    if target_week == "Next Week":
        latest_data = store_sku_data.tail(1).copy()
        target_month = None
        target_weekofyear = None
        target_quarter = None
        
        if 'week' in df_features.columns:
            global_latest_week = pd.to_datetime(df_features['week'].max())
            next_week_date = global_latest_week + pd.Timedelta(days=7)
            target_month = next_week_date.month
            target_weekofyear = next_week_date.isocalendar().week
            target_quarter = next_week_date.quarter
    else:
        latest_data = store_sku_data[store_sku_data['week'] == target_week].copy()
        target_month = None
        target_weekofyear = None
        target_quarter = None
        if latest_data.empty: return None

    old_net_price = latest_data['net_price'].values[0]
    old_discount = latest_data['discount_pct'].values[0]
    cost_price = latest_data['cost_price'].values[0]
    base_price = old_net_price / (1 - old_discount/100) if old_discount != 100 else old_net_price

    test_discounts = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    sim_df = pd.concat([latest_data]*len(test_discounts), ignore_index=True)
    sim_df['discount_pct'] = test_discounts

    if target_month is not None:
        sim_df['month'] = target_month
    if target_weekofyear is not None:
        sim_df['weekofyear'] = target_weekofyear
    if target_quarter is not None:
        sim_df['quarter'] = target_quarter

    sim_df['pred_qty'] = predict_sales(model, sim_df, feature_cols)
    sim_df['sim_net_price'] = base_price * (1 - sim_df['discount_pct'] / 100.0)
    sim_df['expected_revenue'] = sim_df['pred_qty'] * sim_df['sim_net_price']
    sim_df['expected_profit'] = sim_df['pred_qty'] * (sim_df['sim_net_price'] - cost_price)

    return sim_df[['discount_pct', 'sim_net_price', 'pred_qty', 'expected_revenue', 'expected_profit']]

def run_global_optimization(df_features, model, feature_cols, objective="profit"):
    """Tối ưu hóa chiến lược giá cho toàn bộ hệ thống."""
    latest_idx = df_features.groupby(['store_id', 'sku_id'])['week'].idxmax()
    latest_data = df_features.loc[latest_idx].copy()

    latest_data['base_price'] = np.where(
        latest_data['discount_pct'] != 100,
        latest_data['net_price'] / (1 - latest_data['discount_pct'] / 100.0),
        latest_data['net_price']
    )

    test_discounts = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    scenarios = []
    for d in test_discounts:
        temp = latest_data.copy()
        temp['discount_pct'] = d
        temp['sim_net_price'] = temp['base_price'] * (1 - d / 100.0)
        scenarios.append(temp)

    df_sim = pd.concat(scenarios, ignore_index=True)
    df_sim['pred_qty'] = predict_sales(model, df_sim, feature_cols)
    df_sim['expected_revenue'] = df_sim['pred_qty'] * df_sim['sim_net_price']
    df_sim['expected_profit'] = df_sim['pred_qty'] * (df_sim['sim_net_price'] - df_sim['cost_price'])

    if objective == "profit":
        best_idx = df_sim.groupby(['store_id', 'sku_id'])['expected_profit'].idxmax()
        df_optimal = df_sim.loc[best_idx].copy()
    else: # revenue
        best_idx = df_sim.groupby(['store_id', 'sku_id'])['expected_revenue'].idxmax()
        df_optimal = df_sim.loc[best_idx].copy()

    cols = ['store_id', 'sku_id', 'base_price', 'cost_price', 'discount_pct',
            'sim_net_price', 'pred_qty', 'expected_revenue', 'expected_profit']
    return df_optimal[cols].rename(columns={'discount_pct': 'Optimal_Discount_%'})