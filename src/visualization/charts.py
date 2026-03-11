import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd

# màn hình 1: overview
def plot_sales_trend(df, title_suffix="toàn chuỗi"):
    trend = df.groupby('week').agg({'revenue': 'sum', 'pred_revenue': 'sum'}).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend['week'], y=trend['revenue'], 
        mode='lines+markers', name='Thực tế', 
        line=dict(color='#9CA3AF', width=2), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=trend['week'], y=trend['pred_revenue'], 
        mode='lines', name='Dự báo', 
        line=dict(color='#FF6B3D', dash='dash', width=2)
    ))
    
    fig.update_layout(
        template="plotly_white",
        title=f"Xu hướng doanh thu thực tế và dự báo ({title_suffix})",
        xaxis_title="Thời gian (tuần)",
        yaxis_title="Doanh thu ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_seasonality_dual(df):
    season = df.groupby('month').agg(
        total_qty=('quantity', 'sum'),
        total_rev=('revenue', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()
    season['margin_pct'] = np.where(season['total_rev'] > 0, (season['total_profit'] / season['total_rev']) * 100, 0)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=season['month'], y=season['total_qty'], 
        name='Sản lượng', marker_color='#E5E7EB'
    ))
    fig.add_trace(go.Scatter(
        x=season['month'], y=season['margin_pct'], 
        mode='lines+markers', name='Biên lợi nhuận', 
        yaxis='y2', line=dict(color='#FF6B3D', width=3), marker=dict(size=8)
    ))
    
    fig.update_layout(
        template="plotly_white",
        title="Tính mùa vụ: Sản lượng và hiệu quả lợi nhuận",
        xaxis=dict(title="Tháng", tickmode='linear'),
        yaxis=dict(title="Sản lượng", showgrid=True, gridcolor='#F3F4F6'),
        yaxis2=dict(title="Biên lợi nhuận (%)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_top_stores_donut(df):
    store_rev = df.groupby('store_id')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
    top_5 = store_rev.head(5).copy()
    top_5['store_name'] = 'Cửa hàng ' + top_5['store_id'].astype(str)
    
    others_rev = store_rev.iloc[5:]['revenue'].sum()
    if others_rev > 0:
        others_df = pd.DataFrame([{'store_id': 'Others', 'revenue': others_rev, 'store_name': 'Các cửa hàng khác'}])
        top_5 = pd.concat([top_5, others_df], ignore_index=True)
        
    fig = px.pie(
        top_5, values='revenue', names='store_name', hole=0.5, 
        title="Tỷ trọng doanh thu top 5 cửa hàng",
        color_discrete_sequence=['#FF6B3D', '#FDBA74', '#FCA5A5', '#93C5FD', '#D1D5DB', '#F3F4F6']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# màn hình 2: analytics
def plot_discount_efficiency(df):
    df['discount_bin'] = (df['discount_pct'] // 5) * 5
    efficiency = df.groupby('discount_bin').agg(
        avg_profit=('profit', 'mean'), tx_count=('quantity', 'count')
    ).reset_index()
    
    efficiency = efficiency[efficiency['tx_count'] > 10]
    efficiency['discount_label'] = efficiency['discount_bin'].astype(str) + '%'
    efficiency['color'] = np.where(efficiency['avg_profit'] < 0, '#EF4444', '#FF6B3D')
    
    fig = go.Figure(data=[go.Bar(
        x=efficiency['discount_label'], y=efficiency['avg_profit'], marker_color=efficiency['color']
    )])
    
    fig.update_layout(
        template="plotly_white",
        title="Điểm ngoặt khuyến mãi: Lợi nhuận trung bình theo mức giảm",
        xaxis_title="Mức giảm giá",
        yaxis_title="Lợi nhuận trung bình ($)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="Hòa vốn")
    return fig

def plot_store_quadrants(df):
    store_perf = df.groupby('store_id').agg(total_rev=('revenue', 'sum'), total_profit=('profit', 'sum')).reset_index()
    med_rev, med_profit = store_perf['total_rev'].median(), store_perf['total_profit'].median()
    
    fig = px.scatter(
        store_perf, x='total_rev', y='total_profit', text='store_id', 
        size='total_rev', color_discrete_sequence=['#3B82F6'],
        title="Ma trận phân loại cửa hàng",
        labels={'total_rev': 'Tổng doanh thu ($)', 'total_profit': 'Tổng lợi nhuận ($)'}
    )
    
    fig.update_traces(textposition='top center', marker=dict(opacity=0.7, line=dict(color='white', width=1)))
    fig.update_layout(
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6'),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6')
    )
    
    fig.add_vline(x=med_rev, line_dash="dash", line_color="#9CA3AF", opacity=0.8)
    fig.add_hline(y=med_profit, line_dash="dash", line_color="#9CA3AF", opacity=0.8)
    fig.add_annotation(x=store_perf['total_rev'].max(), y=store_perf['total_profit'].max(), text="Ngôi sao", showarrow=False, opacity=0.4, font=dict(size=18, color="#22C55E"))
    fig.add_annotation(x=store_perf['total_rev'].max(), y=store_perf['total_profit'].min(), text="Đốt tiền", showarrow=False, opacity=0.4, font=dict(size=18, color="#EF4444"))
    
    return fig

# màn hình 3 & 4: what-if & optimization
def plot_whatif_curves(res_df, best_discount, best_profit):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res_df['discount_pct'], y=res_df['pred_qty'], 
        name='Sản lượng', yaxis='y2', opacity=0.3, marker_color='#9CA3AF'
    ))
    fig.add_trace(go.Scatter(
        x=res_df['discount_pct'], y=res_df['expected_profit'], 
        mode='lines+markers', name='Lợi nhuận', 
        line=dict(color='#22C55E', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[best_discount], y=[best_profit], mode='markers', 
        marker=dict(color='#FF6B3D', size=14, line=dict(color='white', width=2)), 
        name='Điểm tối ưu'
    ))
    
    fig.update_layout(
        template="plotly_white",
        title="Đường cong phản ứng lợi nhuận và sức mua", 
        xaxis=dict(title="Mức giảm giá (%)", showgrid=True, gridcolor='#F3F4F6'),
        yaxis=dict(title=dict(text="Lợi nhuận ($)", font=dict(color="#22C55E")), tickfont=dict(color="#22C55E"), showgrid=True, gridcolor='#F3F4F6'),
        yaxis2=dict(title=dict(text="Số lượng", font=dict(color="#9CA3AF")), tickfont=dict(color="#9CA3AF"), overlaying="y", side="right"),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_discount_strategy_distribution(opt_df):
    conditions = [
        (opt_df['Optimal_Discount_%'] == 0),
        (opt_df['Optimal_Discount_%'] > 0) & (opt_df['Optimal_Discount_%'] <= 15),
        (opt_df['Optimal_Discount_%'] > 15)
    ]
    choices = ['Giữ nguyên giá (0%)', 'Khuyến mãi nhẹ (5-15%)', 'Xả kho sâu (>15%)']
    
    temp_df = opt_df.copy()
    temp_df['Strategy'] = np.select(conditions, choices, default='Khác')
    
    dist = temp_df['Strategy'].value_counts().reset_index()
    dist.columns = ['Chiến lược', 'Số lượng']
    
    fig = px.pie(
        dist, values='Số lượng', names='Chiến lược', hole=0.45, 
        title="Phân bổ chiến lược giá",
        color='Chiến lược',
        color_discrete_map={
            'Giữ nguyên giá (0%)': '#3B82F6',
            'Khuyến mãi nhẹ (5-15%)': '#22C55E',
            'Xả kho sâu (>15%)': '#EF4444'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# màn hình 5: shap
def plot_shap_summary(model, df_features, feature_cols):
    feature_mapping = {
        "discount_pct": "Mức khuyến mãi (%)",
        "rolling_mean_4": "Sức mua trung bình (tháng)",
        "rolling_mean_2": "Sức mua trung bình (nửa tháng)",
        "lag_1": "Doanh số tuần trước",
        "lag_2": "Doanh số 2 tuần trước",
        "lag_4": "Doanh số 4 tuần trước",
        "store_weekly_sales": "Sức bán cửa hàng (tuần trước)",
        "sku_weekly_sales": "Độ hot sản phẩm (tuần trước)",
        "month": "Tháng",
        "weekofyear": "Tuần trong năm",
        "quarter": "Quý",
        "sku_id": "Mã sản phẩm",
        "store_id": "Mã cửa hàng"
    }
    
    X_display = df_features[feature_cols].rename(columns=feature_mapping)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features[feature_cols])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bg_color = '#FFFFFF' 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    shap.summary_plot(shap_values, X_display, show=False)
    
    text_color = '#4B5563'
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.tick_params(colors=text_color, labelsize=11)
    
    cb = plt.gcf().axes[-1] 
    cb.tick_params(colors=text_color)
    cb.set_ylabel('Giá trị đặc trưng (cao - thấp)', color=text_color, fontsize=12)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    return fig