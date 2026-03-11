import numpy as np

def predict_sales(model, df_features, feature_cols):
    """Hàm lõi chuyên biệt để chạy suy luận (Inference)."""
    pred_log = model.predict(df_features[feature_cols])
    pred_qty = np.clip(np.round(np.expm1(pred_log)), 0, None)
    return pred_qty