import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd

# ==========================================
# MÀN HÌNH 1: TỔNG QUAN (OVERVIEW)
# ==========================================
def plot_sales_trend(df, title_suffix="Toàn chuỗi"):
    """Biểu đồ xu hướng kết hợp Thực tế và Dự báo."""
    trend = df.groupby('week').agg({
        'revenue': 'sum', 
        'pred_revenue': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    # Đường Thực tế (Màu xanh dương)
    fig.add_trace(go.Scatter(
        x=trend['week'], y=trend['revenue'], 
        mode='lines+markers', name='Thực tế (Actual)', 
        line=dict(color='dodgerblue', width=2)
    ))
    # Đường Dự báo (Màu cam nét đứt)
    fig.add_trace(go.Scatter(
        x=trend['week'], y=trend['pred_revenue'], 
        mode='lines', name='AI Dự báo (Predicted)', 
        line=dict(color='darkorange', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"📈 Xu hướng Doanh thu Thực tế vs Dự báo ({title_suffix})",
        xaxis_title="Thời gian (Tuần)",
        yaxis_title="Doanh thu (VNĐ)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_seasonality_dual(df):
    """Tính mùa vụ: Trục chính là Sản lượng, Trục phụ là Biên Lợi Nhuận."""
    season = df.groupby('month').agg(
        total_qty=('quantity', 'sum'),
        total_rev=('revenue', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()
    
    # Tránh chia cho 0
    season['margin_pct'] = np.where(season['total_rev'] > 0, (season['total_profit'] / season['total_rev']) * 100, 0)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=season['month'], y=season['total_qty'], name='Sản lượng (Qty)', opacity=0.7))
    fig.add_trace(go.Scatter(x=season['month'], y=season['margin_pct'], mode='lines+markers', name='Biên Lợi nhuận (%)', yaxis='y2', line=dict(width=3)))
    
    fig.update_layout(
        title="🗓️ Tính mùa vụ: Sản lượng vs Hiệu quả Lợi nhuận",
        xaxis=dict(title="Tháng", tickmode='linear'),
        yaxis=dict(title="Sản lượng"),
        yaxis2=dict(title="Biên Lợi nhuận (%)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified"
    )
    return fig

def plot_top_stores_donut(df):
    """Tỷ trọng doanh thu của Top 5 cửa hàng gánh team."""
    store_rev = df.groupby('store_id')['revenue'].sum().reset_index()
    store_rev = store_rev.sort_values('revenue', ascending=False)
    
    # Lấy Top 5, phần còn lại gom vào "Others"
    top_5 = store_rev.head(5).copy()
    top_5['store_name'] = 'Store ' + top_5['store_id'].astype(str)
    
    others_rev = store_rev.iloc[5:]['revenue'].sum()
    if others_rev > 0:
        others_df = pd.DataFrame([{'store_id': 'Others', 'revenue': others_rev, 'store_name': 'Các Store Khác'}])
        top_5 = pd.concat([top_5, others_df], ignore_index=True)
        
    fig = px.pie(top_5, values='revenue', names='store_name', hole=0.4, title="🏆 Tỷ trọng Doanh thu Top 5 Cửa hàng")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# ==========================================
# MÀN HÌNH 2: PHÂN TÍCH (ANALYTICS)
# ==========================================
def plot_discount_efficiency(df):
    """Hiệu quả Khuyến mãi: Mức giảm giá nào đem lại Lợi nhuận trung bình cao nhất?"""
    # Gom nhóm các mức discount
    df['discount_bin'] = (df['discount_pct'] // 5) * 5
    efficiency = df.groupby('discount_bin').agg(
        avg_profit=('profit', 'mean'),
        tx_count=('quantity', 'count')
    ).reset_index()
    
    efficiency = efficiency[efficiency['tx_count'] > 10] # Chỉ lấy các nhóm có ý nghĩa thống kê
    efficiency['discount_label'] = efficiency['discount_bin'].astype(str) + '%'
    
    fig = px.bar(efficiency, x='discount_label', y='avg_profit', 
                 title="🎯 Điểm ngoặt Khuyến mãi: Lợi nhuận Trung bình theo Mức giảm giá",
                 labels={'discount_label': 'Mức Giảm Giá', 'avg_profit': 'Lợi Nhuận Trung Bình / Giao dịch'})
    # Thêm đường số 0 để thấy rõ mốc bị lỗ
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Điểm Hòa Vốn")
    return fig

def plot_store_quadrants(df):
    """Ma trận Hiệu suất Cửa hàng (BCG Matrix style)."""
    store_perf = df.groupby('store_id').agg(
        total_rev=('revenue', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()
    
    med_rev = store_perf['total_rev'].median()
    med_profit = store_perf['total_profit'].median()
    
    fig = px.scatter(store_perf, x='total_rev', y='total_profit', text='store_id', size='total_rev',
                     title="🧭 Ma trận Sức khỏe Cửa hàng (Doanh thu vs Lợi nhuận)",
                     labels={'total_rev': 'Tổng Doanh Thu', 'total_profit': 'Tổng Lợi Nhuận'})
    
    fig.update_traces(textposition='top center', marker=dict(opacity=0.7))
    fig.add_vline(x=med_rev, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=med_profit, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Gắn nhãn 4 góc phần tư
    fig.add_annotation(x=store_perf['total_rev'].max(), y=store_perf['total_profit'].max(), text="Ngôi Sao (Tốt)", showarrow=False, opacity=0.3, font=dict(size=20))
    fig.add_annotation(x=store_perf['total_rev'].max(), y=store_perf['total_profit'].min(), text="Đốt Tiền", showarrow=False, opacity=0.3, font=dict(size=20))
    
    return fig

# --- WHAT-IF CHARTS ---
def plot_whatif_curves(res_df, best_discount, best_profit):
    fig = go.Figure()
    
    # Cột Số lượng (Trục Y2)
    fig.add_trace(go.Bar(
        x=res_df['discount_pct'], y=res_df['pred_qty'], 
        name='Số lượng (Qty)', yaxis='y2', opacity=0.4, marker_color='dodgerblue'
    ))
    # Đường Lợi nhuận (Trục Y chính)
    fig.add_trace(go.Scatter(
        x=res_df['discount_pct'], y=res_df['expected_profit'], 
        mode='lines+markers', name='Lợi nhuận (Profit)', 
        line=dict(color='firebrick', width=3)
    ))
    # Điểm tối ưu
    fig.add_trace(go.Scatter(
        x=[best_discount], y=[best_profit], mode='markers', 
        marker=dict(color='gold', size=15, line=dict(color='black', width=2)), 
        name='Điểm Tối Ưu'
    ))
    
    # Cập nhật Syntax chuẩn cho Plotly 5+
    fig.update_layout(
        title="Đường cong Phản ứng Lợi nhuận & Sức mua", 
        xaxis=dict(title="Mức Giảm Giá (%)"),
        yaxis=dict(
            title=dict(text="Lợi nhuận", font=dict(color="firebrick")),
            tickfont=dict(color="firebrick")
        ),
        yaxis2=dict(
            title=dict(text="Số lượng", font=dict(color="dodgerblue")),
            tickfont=dict(color="dodgerblue"),
            overlaying="y", 
            side="right"
        ),
        hovermode="x unified"
    )
    return fig

# --- XAI (SHAP) CHARTS ---
def plot_shap_summary(model, df_features, feature_cols):
    """Biểu đồ SHAP chuẩn Dark Mode và tên biến thuần Việt."""
    import matplotlib.pyplot as plt
    
    # 1. Từ điển dịch tên biến sang ngôn ngữ Kinh doanh
    feature_mapping = {
        "discount_pct": "Mức Khuyến Mãi (%)",
        "rolling_mean_4": "Sức Mua TB (Tháng)",
        "rolling_mean_2": "Sức Mua TB (Nửa Tháng)",
        "lag_1": "Doanh Số Tuần Trước",
        "lag_2": "Doanh Số 2 Tuần Trước",
        "lag_4": "Doanh Số 4 Tuần Trước",
        "store_weekly_sales": "Sức Bán Cửa Hàng (Tuần Trước)",
        "sku_weekly_sales": "Độ Hot Sản Phẩm (Tuần Trước)",
        "month": "Tháng",
        "weekofyear": "Tuần trong Năm",
        "quarter": "Quý",
        "sku_id": "Mã Sản Phẩm",
        "store_id": "Mã Cửa Hàng"
    }
    
    # Đổi tên cột để hiển thị lên biểu đồ
    X_display = df_features[feature_cols].rename(columns=feature_mapping)
    
    # 2. Tính toán SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features[feature_cols])
    
    # 3. Ép giao diện Dark Mode cho Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Đặt màu nền trùng với màu nền mặc định của Streamlit (Màu xám đen than)
    bg_color = '#0e1117' 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Vẽ biểu đồ
    shap.summary_plot(shap_values, X_display, show=False)
    
    # Chuyển đổi màu chữ các trục sang màu Trắng/Xám sáng
    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.tick_params(colors='lightgray', labelsize=11)
    
    # Chỉnh màu thanh Colorbar bên phải
    cb = plt.gcf().axes[-1] 
    cb.tick_params(colors='lightgray')
    cb.set_ylabel('Giá trị Đặc trưng (Cao - Thấp)', color='lightgray', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_discount_strategy_distribution(opt_df):
    """Biểu đồ phân phối chiến lược giá đề xuất toàn hệ thống."""
    # Phân nhóm chiến lược giá
    conditions = [
        (opt_df['Optimal_Discount_%'] == 0),
        (opt_df['Optimal_Discount_%'] > 0) & (opt_df['Optimal_Discount_%'] <= 15),
        (opt_df['Optimal_Discount_%'] > 15)
    ]
    choices = ['Giữ nguyên giá (0%)', 'Khuyến mãi nhẹ (5-15%)', 'Xả kho sâu (>15%)']
    
    # Tạo bản sao để không ảnh hưởng data gốc
    temp_df = opt_df.copy()
    temp_df['Strategy'] = np.select(conditions, choices, default='Khác')
    
    dist = temp_df['Strategy'].value_counts().reset_index()
    dist.columns = ['Chiến lược', 'Số lượng SKU']
    
    fig = px.pie(dist, values='Số lượng SKU', names='Chiến lược', hole=0.4, 
                 title="🍩 Phân bổ Chiến lược Giá Toàn Hệ thống",
                 color='Chiến lược',
                 color_discrete_map={
                     'Giữ nguyên giá (0%)': 'dodgerblue',
                     'Khuyến mãi nhẹ (5-15%)': 'mediumseagreen',
                     'Xả kho sâu (>15%)': 'firebrick'
                 })
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig