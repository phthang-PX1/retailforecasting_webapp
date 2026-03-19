import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.models.predict import predict_sales
from src.optimization.discount_optimizer import simulate_what_if, run_global_optimization
from src.visualization import charts

# cấu hình giao diện
st.set_page_config(page_title="Retail pricing dashboard", layout="wide")

# tách rời css để dễ quản lý
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 1. Nền trang xám nhạt giúp các thẻ trắng nổi bật 3D */
    .stApp {
        background-color: #F3F4F6 !important;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* 2. Thiết kế Card: KPI, Khung biểu đồ, Expander, Bảng */
    [data-testid="stMetric"], 
    [data-testid="stVerticalBlockBorderWrapper"], 
    div[data-testid="stExpander"], 
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important; /* Bo góc nhỏ lại cho thanh lịch, chuyên nghiệp */
        /* Bóng đổ chuẩn Tailwind CSS (shadow-md) sắc nét và có chiều sâu */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1) !important;
        border: none !important; /* Bỏ hẳn viền xám để bóng làm nhiệm vụ tạo khối */
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 24px !important;
    }

    /* 3. Tinh chỉnh riêng cho khoảng cách bên trong thẻ KPI */
    [data-testid="stMetric"] {
        padding: 20px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important; color: #6B7280 !important; font-weight: 500; margin-bottom: 8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important; color: #111827 !important; font-weight: 700;
    }

    /* 4. Phân cấp tiêu đề */
    h1 { font-size: 28px !important; font-weight: 700 !important; color: #111827; margin-bottom: 24px;}
    h3 { font-size: 18px !important; font-weight: 600 !important; color: #111827; margin-bottom: 16px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 1. tải tài nguyên và chuẩn bị dữ liệu
@st.cache_resource
def load_system_assets():
    model = joblib.load('models/lightgbm_discount_model.pkl')
    features_df = pd.read_csv('data/features/model_features.csv')
    features_df['week'] = pd.to_datetime(features_df['week'])
    return model, features_df

model, df = load_system_assets()

feature_cols = [
    "store_id", "sku_id", "discount_pct", "weekofyear", "month",
    "quarter", "lag_1", "lag_2", "lag_4", "rolling_mean_2",
    "rolling_mean_4", "store_weekly_sales", "sku_weekly_sales"
]

# bổ sung các cột dữ liệu tính toán
if 'pred_qty' not in df.columns:
    df['pred_qty'] = predict_sales(model, df, feature_cols)

df['revenue'] = df['quantity'] * df['net_price']
df['profit'] = df['quantity'] * (df['net_price'] - df['cost_price'])
df['pred_revenue'] = df['pred_qty'] * df['net_price']

def generate_forecast_for_df(df_trend, model, feature_cols, weeks_to_predict=2):
    last_week = df_trend['week'].max()
    latest_data = df_trend[df_trend['week'] == last_week].copy()
    
    future_dfs = []
    curr_data = latest_data.copy()
    
    for i in range(1, weeks_to_predict + 1):
        next_week = last_week + pd.Timedelta(days=7*i)
        next_data = curr_data.copy()
        
        next_data['week'] = next_week
        next_data['weekofyear'] = next_week.isocalendar().week
        next_data['month'] = next_week.month
        next_data['quarter'] = next_week.quarter
        
        next_data['lag_4'] = next_data['lag_2']
        next_data['lag_2'] = next_data['lag_1']
        next_data['lag_1'] = curr_data['pred_qty'] if 'pred_qty' in curr_data else curr_data['quantity']
        
        next_data['pred_qty'] = predict_sales(model, next_data, feature_cols)
        next_data['pred_revenue'] = next_data['pred_qty'] * next_data['net_price']
        
        next_data['revenue'] = np.nan
        next_data['quantity'] = np.nan
        
        future_dfs.append(next_data)
        curr_data = next_data
        
    if future_dfs:
        return pd.concat([df_trend] + future_dfs, ignore_index=True)
    return df_trend

# 2. thanh điều hướng
st.sidebar.title("Menu điều hướng")
st.sidebar.markdown("Hệ thống hỗ trợ ra quyết định")
page = st.sidebar.radio("Chọn màn hình phân tích:", [
    "1. Tổng quan kinh doanh",
    "2. Phân tích dữ liệu",
    "3. Mô phỏng kịch bản",
    "4. Đề xuất tối ưu",
    "5. Giải thích mô hình"
])

# 3. triển khai các màn hình
if page == "1. Tổng quan kinh doanh":
    st.title("Bảng điều khiển tổng quan")
    
    # lấy số liệu 2 tuần gần nhất
    sorted_weeks = sorted(df['week'].unique())
    latest_week = sorted_weeks[-1] if len(sorted_weeks) > 0 else None
    prev_week = sorted_weeks[-2] if len(sorted_weeks) > 1 else latest_week
        
    curr_df = df[df['week'] == latest_week]
    prev_df = df[df['week'] == prev_week]
    
    curr_rev, prev_rev = curr_df['revenue'].sum(), prev_df['revenue'].sum()
    curr_qty, prev_qty = curr_df['quantity'].sum(), prev_df['quantity'].sum()
    curr_margin = (curr_df['profit'].sum() / curr_rev * 100) if curr_rev > 0 else 0
    prev_margin = (prev_df['profit'].sum() / prev_rev * 100) if prev_rev > 0 else 0
    curr_discount = curr_df['discount_pct'].mean()
    prev_discount = prev_df['discount_pct'].mean()

    st.markdown(f"**Báo cáo tuần: {pd.to_datetime(latest_week).strftime('%d/%m/%Y')}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doanh thu tuần", f"${curr_rev:,.0f}", f"{(curr_rev/prev_rev - 1)*100:.1f}%" if prev_rev else "0%")
    c2.metric("Sản lượng bán", f"{curr_qty:,.0f}", f"{(curr_qty/prev_qty - 1)*100:.1f}%" if prev_qty else "0%")
    c3.metric("Biên lợi nhuận", f"{curr_margin:.1f}%", f"{curr_margin - prev_margin:.1f}%", delta_color="normal")
    c4.metric("Giảm giá trung bình", f"{curr_discount:.1f}%", f"{curr_discount - prev_discount:.1f}%", delta_color="inverse")
    
    st.write("") # tạo khoảng trống nhỏ thay vì dùng divider
    
    # phân tích xu hướng (nằm trong card)
    with st.container(border=True):
        st.markdown("### Phân tích xu hướng doanh thu")
        trend_level = st.radio("Cấp độ phân tích:", ["Toàn hệ thống", "Theo cửa hàng", "Theo sản phẩm", "Theo cửa hàng và sản phẩm"], horizontal=True)
        
        df_trend = df.copy()
        title_suffix = "toàn chuỗi"
        
        if trend_level == "Theo cửa hàng":
            sel_trend_store = st.selectbox("Chọn cửa hàng:", sorted(df['store_id'].unique()))
            df_trend = df_trend[df_trend['store_id'] == sel_trend_store]
            title_suffix = f"cửa hàng {sel_trend_store}"
        elif trend_level == "Theo sản phẩm":
            sel_trend_sku = st.selectbox("Chọn sản phẩm (SKU):", sorted(df['sku_id'].unique()))
            df_trend = df_trend[df_trend['sku_id'] == sel_trend_sku]
            title_suffix = f"sản phẩm {sel_trend_sku}"
        elif trend_level == "Theo cửa hàng và sản phẩm":
            c1, c2 = st.columns(2)
            sel_trend_store = c1.selectbox("Chọn cửa hàng:", sorted(df['store_id'].unique()))
            valid_skus = sorted(df[df['store_id'] == sel_trend_store]['sku_id'].unique())
            sel_trend_sku = c2.selectbox("Chọn sản phẩm (SKU):", valid_skus)
            df_trend = df_trend[(df_trend['store_id'] == sel_trend_store) & (df_trend['sku_id'] == sel_trend_sku)]
            title_suffix = f"cửa hàng {sel_trend_store} - SKU {sel_trend_sku}"
            
        df_trend_forecast = generate_forecast_for_df(df_trend, model, feature_cols, weeks_to_predict=2)
            
        st.plotly_chart(charts.plot_sales_trend(df_trend_forecast, title_suffix), use_container_width=True)
    
    st.write("")
    
    # 2 biểu đồ dưới cùng (nằm trong 2 card song song)
    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        with st.container(border=True):
            st.plotly_chart(charts.plot_seasonality_dual(df), use_container_width=True)
    with col_chart2:
        with st.container(border=True):
            st.plotly_chart(charts.plot_top_stores_bar(df), use_container_width=True)

elif page == "2. Phân tích dữ liệu":
    st.title("Phân tích bán hàng và nhu cầu")
    
    # logic phân lớp cửa hàng
    loss_df = df[df['profit'] < 0]
    profit_leakage = abs(loss_df['profit'].sum())
    
    discount_perf = df.groupby('discount_pct').agg({'profit': 'mean', 'quantity': 'sum'}).reset_index()
    best_discount_pct = discount_perf.loc[discount_perf['profit'].idxmax()]['discount_pct']
    
    store_perf = df.groupby('store_id').agg(revenue=('revenue', 'sum'), profit=('profit', 'sum')).reset_index()
    med_rev, med_profit = store_perf['revenue'].median(), store_perf['profit'].median()
    
    conditions = [
        (store_perf['revenue'] > med_rev) & (store_perf['profit'] > med_profit),
        (store_perf['revenue'] > med_rev) & (store_perf['profit'] <= med_profit),
        (store_perf['revenue'] <= med_rev) & (store_perf['profit'] > med_profit),
        (store_perf['revenue'] <= med_rev) & (store_perf['profit'] <= med_profit)
    ]
    choices = ['Ngôi sao', 'Đốt tiền khuyến mãi', 'Tiềm năng tăng trưởng', 'Hiệu suất thấp']
    store_perf['segment'] = np.select(conditions, choices, default='Khác')
    
    st.markdown("### Tóm tắt phát hiện và cảnh báo")
    insight_col, alert_col = st.columns([2, 1])
    
    with insight_col:
        st.info(f"""
        **Các phát hiện quan trọng từ dữ liệu:**
        * Khoảng giảm giá tối ưu: Lợi nhuận trung bình đạt đỉnh ở mức giảm giá **{best_discount_pct:.0f}%**.
        * Rủi ro lợi nhuận: Giảm giá vượt mốc 30% làm biên lợi nhuận sụt giảm mạnh.
        * Hiệu suất: Có **{len(store_perf[store_perf['segment'] == 'Đốt tiền khuyến mãi'])}** cửa hàng doanh thu cao nhưng lợi nhuận kém do lạm dụng khuyến mãi.
        """)
    with alert_col:
        st.error(f"""
        **Chỉ báo thất thoát lợi nhuận:**
        \n# ${profit_leakage:,.0f}
        \n*(Ước tính giá trị mất đi do bán dưới giá vốn trong lịch sử)*
        """)

    st.divider()
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.plotly_chart(charts.plot_discount_efficiency(df), use_container_width=True)
    with col_a2:
        st.plotly_chart(charts.plot_store_quadrants(df), use_container_width=True)

    st.divider()
    st.markdown("### Phân loại cửa hàng và đề xuất")
    seg_counts = store_perf['segment'].value_counts()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ngôi sao", f"{seg_counts.get('Ngôi sao', 0)} cửa hàng", "Duy trì", delta_color="normal")
    c2.metric("Đốt tiền khuyến mãi", f"{seg_counts.get('Đốt tiền khuyến mãi', 0)} cửa hàng", "Giảm chiết khấu", delta_color="inverse")
    c3.metric("Tiềm năng tăng trưởng", f"{seg_counts.get('Tiềm năng tăng trưởng', 0)} cửa hàng", "Đẩy số lượng", delta_color="normal")
    c4.metric("Hiệu suất thấp", f"{seg_counts.get('Hiệu suất thấp', 0)} cửa hàng", "Đánh giá lại", delta_color="off")

    with st.expander("Xem chi tiết chiến lược cho từng nhóm", expanded=True):
        st.markdown("""
        * **Ngôi sao (doanh thu cao, lợi nhuận cao):** Duy trì mức giá hiện tại. Định chuẩn chương trình khuyến mãi cho các cửa hàng khác.
        * **Đốt tiền khuyến mãi (doanh thu cao, lợi nhuận thấp):** Cắt giảm khuyến mãi sâu. Chấp nhận giảm số lượng để phục hồi biên lợi nhuận.
        * **Tiềm năng tăng trưởng (doanh thu thấp, lợi nhuận cao):** Thiết kế gói sản phẩm hoặc giảm giá nhẹ để thu hút thêm khách hàng.
        * **Hiệu suất thấp (doanh thu thấp, lợi nhuận thấp):** Kiểm soát chặt tồn kho, đánh giá lại cơ cấu hàng hóa tại điểm bán.
        """)

    st.divider()
    st.markdown("### Top cơ hội cải thiện")
    action_df = store_perf[store_perf['segment'] == 'Đốt tiền khuyến mãi'].sort_values('revenue', ascending=False)
    
    if len(action_df) > 0:
        action_df['đề xuất hành động'] = "Giảm mức chiết khấu, rà soát chiến dịch xả kho."
        st.dataframe(
            action_df[['store_id', 'revenue', 'profit', 'đề xuất hành động']].style.format({
                'revenue': '${:,.0f}', 'profit': '${:,.0f}'
            }).background_gradient(subset=['profit'], cmap='Reds_r'),
            use_container_width=True, hide_index=True
        )
    else:
        st.success("Hệ thống ổn định: Không có cửa hàng nào rơi vào nhóm đốt tiền khuyến mãi.")

elif page == "3. Mô phỏng kịch bản":
    st.title("Mô phỏng kịch bản giảm giá")
    st.markdown("Giả lập thay đổi sức mua và lợi nhuận khi điều chỉnh mức giảm giá cho từng sản phẩm.")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    sel_store = col_sel1.selectbox("Chọn cửa hàng", sorted(df['store_id'].unique()))
    sel_sku = col_sel2.selectbox("Chọn sản phẩm", sorted(df[df['store_id'] == sel_store]['sku_id'].unique()))
    
    store_sku_df = df[(df['store_id'] == sel_store) & (df['sku_id'] == sel_sku)]
    available_weeks = sorted(store_sku_df['week'].unique(), reverse=True)
    
    # Định dạng hiển thị dropdown
    time_options = ["Tuần tiếp theo (Dự báo)"] + [pd.to_datetime(w).strftime('%d/%m/%Y') for w in available_weeks]
    
    target_week_str = col_sel3.selectbox("Chọn thời điểm mô phỏng", time_options)
    
    if target_week_str == "Tuần tiếp theo (Dự báo)":
        target_week_val = "Next Week"
    else:
        # Chuyển ngược từ DD/MM/YYYY về ngảy gốc tương ứng
        selected_idx = time_options.index(target_week_str) - 1
        target_week_val = available_weeks[selected_idx]
    
    if st.button("Chạy kịch bản mô phỏng", type="primary"):
        res_df = simulate_what_if(sel_store, sel_sku, df, model, feature_cols, target_week=target_week_val)
        
        if res_df is not None:
            if target_week_val == "Next Week":
                latest_data = store_sku_df.sort_values('week').tail(1).copy()
                global_max_date = df['week'].max()
                sim_date_obj = pd.to_datetime(global_max_date) + pd.Timedelta(days=7)
                sim_date_str = sim_date_obj.strftime('%d/%m/%Y') + " (Dự báo)"
                curr_discount, curr_profit, curr_qty = latest_data['discount_pct'].values[0], latest_data['profit'].values[0], latest_data['quantity'].values[0]
            else:
                latest_data = store_sku_df[store_sku_df['week'] == target_week_val].copy()
                sim_date_raw = target_week_val
                sim_date_str = pd.to_datetime(sim_date_raw).strftime('%d/%m/%Y') + " (Lịch sử)"
                curr_discount, curr_profit, curr_qty = latest_data['discount_pct'].values[0], latest_data['profit'].values[0], latest_data['quantity'].values[0]
            
            best_idx = res_df['expected_profit'].idxmax()
            best_discount, best_profit, best_qty = res_df.loc[best_idx, 'discount_pct'], res_df.loc[best_idx, 'expected_profit'], res_df.loc[best_idx, 'pred_qty']
            
            st.markdown("### So sánh hiệu quả: hiện tại và đề xuất")
            
            if best_profit < 0:
                st.warning("Cảnh báo xả kho: Mức giá đề xuất dẫn đến bán dưới giá vốn, chỉ phù hợp mục đích thanh lý tồn kho.")
            else:
                st.success(f"Khuyến nghị: Điều chỉnh mức giảm giá từ {curr_discount:.0f}% thành {best_discount:.0f}% để tối ưu lợi nhuận.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Lợi nhuận kỳ vọng", f"${best_profit:,.0f}", f"${(best_profit - curr_profit):,.0f} so với hiện tại")
            c2.metric("Sản lượng kỳ vọng", f"{best_qty:,.0f}", f"{(best_qty - curr_qty):,.0f} so với hiện tại")
            c3.metric("Mức giảm giá đề xuất", f"{best_discount:.0f}%", f"{(best_discount - curr_discount):.0f}% so với hiện tại", delta_color="inverse")
            
            st.divider()
            st.plotly_chart(charts.plot_whatif_curves(res_df, best_discount, best_profit, sim_date=sim_date_str), use_container_width=True)
            
            with st.expander("Xem chi tiết các kịch bản giá"):
                st.dataframe(res_df.style.format({
                    "discount_pct": "{:.0f}%", "sim_net_price": "${:,.2f}", "pred_qty": "{:,.0f}",
                    "expected_revenue": "${:,.0f}", "expected_profit": "${:,.0f}"
                }).background_gradient(subset=['expected_profit'], cmap='Greens'), use_container_width=True)
        else:
            st.warning("Hệ thống chưa đủ dữ liệu lịch sử để mô phỏng sản phẩm này.")

elif page == "4. Đề xuất tối ưu":
    st.title("Tối ưu hóa chiến lược giá")
    st.markdown("Hệ thống tự động duyệt qua các kịch bản để lập kế hoạch giá tối ưu toàn chuỗi.")
    
    objective = st.radio("Mục tiêu chiến lược tuần tới:", 
                         ["Tối đa lợi nhuận", "Tối đa doanh thu (mục tiêu xả kho)"], horizontal=True)
    obj_param = "profit" if "lợi nhuận" in objective else "revenue"
    
    if st.button("Lập kế hoạch giá", type="primary"):
        with st.spinner("Hệ thống đang xử lý tối ưu hóa..."):
            opt_df = run_global_optimization(df, model, feature_cols, objective=obj_param)
            st.success("Hoàn tất lập kế hoạch giá.")
            
            st.markdown("### Tóm tắt kế hoạch")
            total_est_profit = opt_df['expected_profit'].sum()
            total_est_revenue = opt_df['expected_revenue'].sum()
            avg_opt_discount = opt_df['Optimal_Discount_%'].mean()
            
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Tổng lợi nhuận dự kiến", f"${total_est_profit:,.0f}")
            col_k2.metric("Tổng doanh thu dự kiến", f"${total_est_revenue:,.0f}")
            col_k3.metric("Mức khuyến mãi trung bình", f"{avg_opt_discount:.1f}%")
            
            st.divider()
            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.plotly_chart(charts.plot_discount_strategy_distribution(opt_df), use_container_width=True)
            with col_p2:
                st.info("""
                **Hướng dẫn đọc biểu đồ phân bổ:**
                * **Giữ nguyên giá (0%):** Sản phẩm có sức mua ổn định, cần bảo toàn biên lợi nhuận.
                * **Khuyến mãi nhẹ (5-15%):** Áp dụng để kích thích nhu cầu, đạt điểm tối ưu lợi nhuận.
                * **Xả kho sâu (>15%):** Xử lý hàng tồn đọng chậm luân chuyển, thu hồi dòng tiền.
                """)
            
            st.divider()
            st.markdown("### Bảng hành động chi tiết")
            highlight_col = 'expected_profit' if obj_param == 'profit' else 'expected_revenue'
            
            st.dataframe(opt_df.style.format({
                "base_price": "${:,.2f}", "cost_price": "${:,.2f}", "sim_net_price": "${:,.2f}",
                "pred_qty": "{:,.0f}", "expected_revenue": "${:,.0f}", "expected_profit": "${:,.0f}"
            }).background_gradient(subset=[highlight_col], cmap='Greens'), height=400, use_container_width=True)
            
            csv_data = opt_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải dữ liệu kế hoạch (CSV)", 
                data=csv_data, 
                file_name=f'pricing_plan_{obj_param}.csv', 
                mime='text/csv'
            )

elif page == "5. Giải thích mô hình":
    st.title("Phân rã quyết định của AI")
    st.markdown("Sử dụng thuật toán SHAP để định lượng mức độ tác động của từng biến số đến kết quả dự báo của mô hình.")
    
    st.divider()
    col_x1, col_x2 = st.columns([1, 3])
    with col_x1:
        st.info("""
        **Hướng dẫn đọc biểu đồ:**
        * Trục dọc thể hiện độ quan trọng của biến từ cao xuống thấp.
        * Màu đỏ biểu thị giá trị đầu vào cao, màu xanh là giá trị thấp.
        * Các điểm lệch về bên phải biểu thị tác động làm tăng sản lượng dự báo.
        """)
    with col_x2:
        with st.spinner("Đang kết xuất biểu đồ SHAP..."):
            sample_df = df.sample(n=min(1500, len(df)), random_state=42)
            st.pyplot(charts.plot_shap_summary(model, sample_df, feature_cols))