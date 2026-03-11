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

st.set_page_config(page_title="Retail Pricing Dashboard", layout="wide", page_icon="🛍️")
# Cấu hình giao diện
st.set_page_config(page_title="Retail Pricing Dashboard", layout="wide", page_icon="🛍️")

# --- CSS INJECTION: GIAO DIỆN POWER BI ---
st.markdown("""
<style>
    /* Bo góc và đổ bóng cho các thẻ Metric (KPI Cards) */
    div[data-testid="metric-container"] {
        background-color: #1e1e2d;
        border-radius: 12px;
        padding: 15px 20px;
        border: 1px solid #2d2d3f;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border: 1px solid #4da6ff;
    }
    /* Đổi màu nhãn (Label) của KPI sang màu xám sáng sang trọng */
    div[data-testid="metric-container"] > div:nth-child(1) {
        color: #a0a5b1 !important;
        font-weight: 600;
        font-size: 14px;
    }
    /* Format tiêu đề các Tabs và Radio buttons */
    div.stRadio > div[role="radiogroup"] > label {
        background-color: #1e1e2d;
        padding: 10px 15px;
        border-radius: 8px;
        margin-right: 10px;
        border: 1px solid #2d2d3f;
    }
</style>
""", unsafe_allow_html=True)
# ----------------------------------------
# 1. TẢI TÀI NGUYÊN VÀ CHUẨN BỊ DỮ LIỆU
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

# Tính toán dự báo cho toàn bộ dữ liệu lịch sử (Fix lỗi thiếu cột)
if 'pred_qty' not in df.columns:
    df['pred_qty'] = predict_sales(model, df, feature_cols)

# Tính các KPI tài chính
df['revenue'] = df['quantity'] * df['net_price']
df['profit'] = df['quantity'] * (df['net_price'] - df['cost_price'])
df['pred_revenue'] = df['pred_qty'] * df['net_price']

# 2. THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
st.sidebar.title("🛒 Menu Điều Hướng")
st.sidebar.markdown("Hệ thống Hỗ trợ Ra Quyết định (DSS)")
page = st.sidebar.radio("Chọn Màn hình Phân tích:", [
    "1. Tổng quan Kinh doanh (Overview)",
    "2. Phân tích Dữ liệu (Analytics)",
    "3. Mô phỏng Kịch bản (What-If)",
    "4. Đề xuất Tối ưu (Optimization)",
    "5. Giải thích Mô hình (XAI)"
])

# 3. TRIỂN KHAI CÁC MÀN HÌNH

if page == "1. Tổng quan Kinh doanh (Overview)":
    st.title("📊 Bảng điều khiển Tổng quan (Dashboard Overview)")
    
    # Lọc dữ liệu tuần hiện tại và tuần trước để tính Delta
    sorted_weeks = sorted(df['week'].unique())
    latest_week = sorted_weeks[-1] if len(sorted_weeks) > 0 else None
    prev_week = sorted_weeks[-2] if len(sorted_weeks) > 1 else latest_week
        
    curr_df = df[df['week'] == latest_week]
    prev_df = df[df['week'] == prev_week]
    
    curr_rev = curr_df['revenue'].sum()
    prev_rev = prev_df['revenue'].sum()
    curr_qty = curr_df['quantity'].sum()
    prev_qty = prev_df['quantity'].sum()
    curr_margin = (curr_df['profit'].sum() / curr_rev * 100) if curr_rev > 0 else 0
    prev_margin = (prev_df['profit'].sum() / prev_rev * 100) if prev_rev > 0 else 0
    curr_discount = curr_df['discount_pct'].mean()
    prev_discount = prev_df['discount_pct'].mean()

    st.markdown(f"**🗓️ Báo cáo tuần: {pd.to_datetime(latest_week).strftime('%d/%m/%Y')}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doanh Thu Tuần", f"{curr_rev:,.0f} VNĐ", f"{(curr_rev/prev_rev - 1)*100:.1f}%" if prev_rev else "0%")
    c2.metric("Sản Lượng Bán", f"{curr_qty:,.0f} SP", f"{(curr_qty/prev_qty - 1)*100:.1f}%" if prev_qty else "0%")
    c3.metric("Biên Lợi Nhuận (Margin)", f"{curr_margin:.1f}%", f"{curr_margin - prev_margin:.1f}%", delta_color="normal")
    c4.metric("Giảm Giá (Trung bình)", f"{curr_discount:.1f}%", f"{curr_discount - prev_discount:.1f}%", delta_color="inverse")
    
    st.divider()
    
    # --- BỘ LỌC XU HƯỚNG 3 CẤP ĐỘ ĐỘC LẬP ---
    st.markdown("### 📈 Phân tích Xu hướng Doanh thu")
    trend_level = st.radio("📍 Cấp độ Phân tích:", ["Toàn Hệ Thống", "Theo Cửa hàng (Store)", "Theo Sản phẩm (SKU)"], horizontal=True)
    
    df_trend = df.copy()
    title_suffix = "Toàn chuỗi"
    
    if trend_level == "Theo Cửa hàng (Store)":
        sel_trend_store = st.selectbox("Chọn Cửa hàng:", sorted(df['store_id'].unique()))
        df_trend = df_trend[df_trend['store_id'] == sel_trend_store]
        title_suffix = f"Cửa hàng {sel_trend_store}"
        
    elif trend_level == "Theo Sản phẩm (SKU)":
        # Lọc SKU độc lập không liên quan đến Store
        sel_trend_sku = st.selectbox("Chọn Sản phẩm (SKU):", sorted(df['sku_id'].unique()))
        df_trend = df_trend[df_trend['sku_id'] == sel_trend_sku]
        title_suffix = f"Sản phẩm {sel_trend_sku} (Toàn bộ cửa hàng)"
        
    st.plotly_chart(charts.plot_sales_trend(df_trend, title_suffix), use_container_width=True)
    
    st.divider()
    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.plotly_chart(charts.plot_seasonality_dual(df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(charts.plot_top_stores_donut(df), use_container_width=True)

elif page == "2. Phân tích Dữ liệu (Analytics)":
    st.title("📈 Phân tích Bán hàng & Nhu cầu (Sales Analytics)")
    
    # ==========================================
    # 1. TÍNH TOÁN CÁC CHỈ SỐ BUSINESS (DATA MANIPULATION)
    # ==========================================
    # Lợi nhuận bị thất thoát (Profit Leakage) - Các giao dịch có profit < 0
    loss_df = df[df['profit'] < 0]
    profit_leakage = abs(loss_df['profit'].sum())
    
    # Tìm khoảng giảm giá tối ưu (Best Discount Range)
    discount_perf = df.groupby('discount_pct').agg({'profit': 'mean', 'quantity': 'sum'}).reset_index()
    best_discount_row = discount_perf.loc[discount_perf['profit'].idxmax()]
    best_discount_pct = best_discount_row['discount_pct']
    
    # Phân loại Cửa hàng (Store Segmentation)
    store_perf = df.groupby('store_id').agg(
        revenue=('revenue', 'sum'),
        profit=('profit', 'sum')
    ).reset_index()
    
    med_rev = store_perf['revenue'].median()
    med_profit = store_perf['profit'].median()
    
    # Gắn nhãn 4 Quadrants
    conditions = [
        (store_perf['revenue'] > med_rev) & (store_perf['profit'] > med_profit),
        (store_perf['revenue'] > med_rev) & (store_perf['profit'] <= med_profit),
        (store_perf['revenue'] <= med_rev) & (store_perf['profit'] > med_profit),
        (store_perf['revenue'] <= med_rev) & (store_perf['profit'] <= med_profit)
    ]
    choices = ['Ngôi sao', 'Đốt tiền khuyến mãi', 'Tiềm năng tăng trưởng', 'Hiệu suất thấp']
    store_perf['segment'] = np.select(conditions, choices, default='Khác')
    
    # ==========================================
    # 2. HIỂN THỊ INSIGHTS & PROFIT LEAKAGE
    # ==========================================
    st.markdown("### 💡 Key Insights & Cảnh báo Thất thoát")
    insight_col, alert_col = st.columns([2, 1])
    
    with insight_col:
        st.info(f"""
        **Tóm tắt Phát hiện Quan trọng (Key Insights):**
        * 🎯 **Khoảng giảm giá tối ưu:** Lợi nhuận trung bình đạt đỉnh ở mức giảm giá **{best_discount_pct:.0f}%**.
        * ⚠️ **Nguy cơ xói mòn lợi nhuận:** Việc giảm giá vượt mốc 30% làm biên lợi nhuận rớt thẳng đứng và dẫn đến bán lỗ.
        * 🏪 **Phân hóa cửa hàng:** Có **{len(store_perf[store_perf['segment'] == 'Đốt tiền khuyến mãi'])}** cửa hàng đang tạo ra doanh thu khủng nhưng mang về lợi nhuận dưới mức trung bình do lạm dụng khuyến mãi.
        """)
        
    with alert_col:
        st.error(f"""
        **🩸 Chỉ báo Thất thoát Lợi nhuận (Profit Leakage):**
        \n# {profit_leakage/1e6:,.0f} Triệu VNĐ
        \n*(Ước tính số tiền bốc hơi do bán dưới giá vốn trong lịch sử)*
        """)

    st.divider()

    # ==========================================
    # 3. BIỂU ĐỒ TRỰC QUAN (ANALYTICS)
    # ==========================================
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.plotly_chart(charts.plot_discount_efficiency(df), use_container_width=True)
    with col_a2:
        st.plotly_chart(charts.plot_store_quadrants(df), use_container_width=True)

    st.divider()

    # ==========================================
    # 4. STORE SEGMENTATION & RECOMMENDED ACTIONS
    # ==========================================
    st.markdown("### 🧭 Phân loại Cửa hàng & Hành động Đề xuất")
    
    # Đếm số lượng cửa hàng mỗi nhóm
    seg_counts = store_perf['segment'].value_counts()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌟 Ngôi Sao", f"{seg_counts.get('Ngôi sao', 0)} Cửa hàng", "Duy trì", delta_color="normal")
    c2.metric("🔥 Đốt tiền Khuyến mãi", f"{seg_counts.get('Đốt tiền khuyến mãi', 0)} Cửa hàng", "Giảm Discount", delta_color="inverse")
    c3.metric("🌱 Tiềm năng Tăng trưởng", f"{seg_counts.get('Tiềm năng tăng trưởng', 0)} Cửa hàng", "Đẩy Volume", delta_color="normal")
    c4.metric("💤 Hiệu suất Thấp", f"{seg_counts.get('Hiệu suất thấp', 0)} Cửa hàng", "Đánh giá lại", delta_color="off")

    # Bản tóm tắt hành động
    with st.expander("📖 Xem chi tiết Chiến lược cho từng nhóm (Recommended Actions)", expanded=True):
        st.markdown("""
        * **🌟 Ngôi Sao (Doanh thu Cao, Lợi nhuận Cao):** Duy trì mức giá hiện tại. Dùng nhóm này làm chuẩn mực (benchmark) để áp dụng chương trình khuyến mãi cho các cửa hàng khác.
        * **🔥 Đốt tiền Khuyến mãi (Doanh thu Cao, Lợi nhuận Thấp):** Cắt giảm ngay các chương trình giảm giá sâu. Chấp nhận hy sinh một phần doanh thu (Volume) để cứu vãn Biên lợi nhuận (Margin).
        * **🌱 Tiềm năng Tăng trưởng (Doanh thu Thấp, Lợi nhuận Cao):** Cửa hàng đang bán rất có lãi nhưng ít người mua. Cần tung ra các gói ưu đãi (Bundle) hoặc giảm giá nhẹ để lôi kéo thêm khách hàng (Tăng Volume).
        * **💤 Hiệu suất Thấp (Doanh thu Thấp, Lợi nhuận Thấp):** Hạn chế tồn kho. Cần phân tích sâu hơn về vị trí địa lý hoặc thay đổi hẳn danh mục sản phẩm (SKU mix) tại đây.
        """)

    st.divider()

    # ==========================================
    # 5. TOP IMPROVEMENT OPPORTUNITIES
    # ==========================================
    st.markdown("### 🚨 Top Cơ hội Cải thiện (Cửa hàng Cần Chấn Chỉnh Ngay)")
    st.caption("Danh sách các cửa hàng nằm trong nhóm 'Đốt tiền Khuyến mãi' - Cần tối ưu lại chính sách giá.")
    
    # Lọc nhóm Đốt tiền, sắp xếp theo Doanh thu từ cao xuống thấp
    action_df = store_perf[store_perf['segment'] == 'Đốt tiền khuyến mãi'].sort_values('revenue', ascending=False)
    
    if len(action_df) > 0:
        action_df['Đề xuất Hành động'] = "Giảm mức discount, rà soát lại các chiến dịch xả kho."
        st.dataframe(
            action_df[['store_id', 'revenue', 'profit', 'Đề xuất Hành động']].style.format({
                'revenue': '{:,.0f} VNĐ', 
                'profit': '{:,.0f} VNĐ'
            }).background_gradient(subset=['profit'], cmap='Reds_r'), # Lợi nhuận càng thấp màu đỏ càng đậm
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("Tín hiệu tốt: Hiện tại không có cửa hàng nào rơi vào trạng thái Đốt tiền khuyến mãi!")

elif page == "3. Mô phỏng Kịch bản (What-If)":
    st.title("⚡ Mô phỏng Kịch bản Giảm giá (What-If Simulation)")
    st.markdown("Giả lập phản ứng của thị trường (Sức mua & Lợi nhuận) khi bạn thay đổi mức giảm giá cho 1 sản phẩm cụ thể.")
    
    col_sel1, col_sel2 = st.columns(2)
    sel_store = col_sel1.selectbox("🏠 Chọn Store", sorted(df['store_id'].unique()))
    sel_sku = col_sel2.selectbox("📦 Chọn SKU", sorted(df[df['store_id'] == sel_store]['sku_id'].unique()))
    
    if st.button("🚀 Chạy Kịch Bản Mô Phỏng", type="primary"):
        res_df = simulate_what_if(sel_store, sel_sku, df, model, feature_cols)
        
        if res_df is not None:
            # 1. Trích xuất Baseline (Thực tế hiện tại)
            latest_data = df[(df['store_id'] == sel_store) & (df['sku_id'] == sel_sku)].sort_values('week').tail(1)
            curr_discount = latest_data['discount_pct'].values[0]
            curr_profit = latest_data['profit'].values[0]
            curr_qty = latest_data['quantity'].values[0]
            
            # 2. Trích xuất Optimal (AI Đề xuất)
            best_idx = res_df['expected_profit'].idxmax()
            best_discount = res_df.loc[best_idx, 'discount_pct']
            best_profit = res_df.loc[best_idx, 'expected_profit']
            best_qty = res_df.loc[best_idx, 'pred_qty']
            
            # 3. Hiển thị So sánh (Current vs AI Recommended)
            st.markdown("### ⚖️ So sánh Hiệu quả: Hiện tại vs Đề xuất AI")
            
            # Cảnh báo nếu AI khuyên xả kho bán lỗ
            if best_profit < 0:
                st.warning("⚠️ **Cảnh báo Xả kho:** Mức giá AI đề xuất sẽ dẫn đến bán lỗ giá vốn. Tuy nhiên đây là cách tốt nhất để đẩy lượng hàng tồn đọng.")
            else:
                st.success(f"💡 **Khuyến nghị từ AI:** Thay đổi mức giảm giá từ **{curr_discount:.0f}%** thành **{best_discount:.0f}%** để tối ưu lợi nhuận.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Lợi nhuận Kỳ vọng (AI)", f"{best_profit:,.0f} VNĐ", f"{(best_profit - curr_profit):,.0f} VNĐ so với hiện tại")
            c2.metric("Sản lượng Kỳ vọng (AI)", f"{best_qty:,.0f} SP", f"{(best_qty - curr_qty):,.0f} SP so với hiện tại")
            c3.metric("Mức Giảm Giá Đề Xuất", f"{best_discount:.0f}%", f"{(best_discount - curr_discount):.0f}% so với hiện tại", delta_color="inverse")
            
            st.divider()
            
            # 4. Biểu đồ Đường cong và Bảng chi tiết
            st.plotly_chart(charts.plot_whatif_curves(res_df, best_discount, best_profit), use_container_width=True)
            
            with st.expander("🔎 Xem Ma trận Kịch bản chi tiết (Scenario Matrix)"):
                st.dataframe(res_df.style.format({
                    "discount_pct": "{:.0f}%", "sim_net_price": "{:,.0f}", "pred_qty": "{:,.0f}",
                    "expected_revenue": "{:,.0f}", "expected_profit": "{:,.0f}"
                }).background_gradient(subset=['expected_profit'], cmap='Greens'), use_container_width=True)
        else:
            st.warning("Không có đủ dữ liệu lịch sử cho sản phẩm này để chạy mô phỏng.")

elif page == "4. Đề xuất Tối ưu (Optimization)":
    st.title("🎯 Master Plan: Tối ưu hóa Toàn Hệ thống")
    st.markdown("Chỉ với 1 cú click, AI sẽ duyệt qua hàng ngàn sản phẩm trên toàn chuỗi và lập ra bảng kế hoạch giá (Pricing Plan) tối ưu nhất cho tuần tới.")
    
    objective = st.radio("📍 Mục tiêu Chiến lược của bạn tuần tới là gì?", 
                         ["💰 Tối đa Lợi Nhuận (Profit Maximization)", "🔥 Đẩy Doanh Thu / Xả kho (Revenue Maximization)"], 
                         horizontal=True)
    obj_param = "profit" if "Profit" in objective else "revenue"
    
    if st.button("🧠 Lập Kế hoạch Giá Hàng Loạt", type="primary"):
        with st.spinner("AI đang giải quyết bài toán tối ưu trên hàng chục ngàn kịch bản... Vui lòng đợi."):
            opt_df = run_global_optimization(df, model, feature_cols, objective=obj_param)
            
            st.success("✅ Đã lập xong Kế hoạch Giá toàn chuỗi!")
            
            # 1. Executive Summary (Báo cáo tóm tắt cho C-Level)
            st.markdown("### 📋 Tóm tắt Kế hoạch (Executive Summary)")
            total_est_profit = opt_df['expected_profit'].sum()
            total_est_revenue = opt_df['expected_revenue'].sum()
            avg_opt_discount = opt_df['Optimal_Discount_%'].mean()
            
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Tổng Lợi nhuận Dự kiến", f"{total_est_profit/1e6:,.1f} Triệu VNĐ")
            col_k2.metric("Tổng Doanh thu Dự kiến", f"{total_est_revenue/1e6:,.1f} Triệu VNĐ")
            col_k3.metric("Mức Khuyến mãi Trung bình", f"{avg_opt_discount:.1f}%")
            
            st.divider()
            
            # 2. Phân bổ chiến lược
            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.plotly_chart(charts.plot_discount_strategy_distribution(opt_df), use_container_width=True)
            with col_p2:
                st.info("""
                **Cách đọc biểu đồ Chiến lược:**
                * **Giữ nguyên giá (0%):** Nhóm sản phẩm đang có sức mua tốt, không cần giảm giá để bảo toàn biên lợi nhuận.
                * **Khuyến mãi nhẹ (5-15%):** Nhóm cần một cú hích nhỏ để kích thích người mua (Tối ưu hóa điểm uốn của cầu).
                * **Xả kho sâu (>15%):** Nhóm sản phẩm ế ẩm, bắt buộc phải cắt máu lợi nhuận để thu hồi vốn và giải phóng không gian lưu trữ.
                """)
            
            st.divider()
            
            # 3. Actionable Table & Export
            st.markdown("### 🛠️ Bảng Hành động Chi tiết (Actionable Pricing List)")
            st.caption("Danh sách mức giá cụ thể cần thiết lập cho hệ thống POS/ERP tuần tới.")
            
            # Tô màu bảng theo objective để người dùng tập trung vào đúng cột
            highlight_col = 'expected_profit' if obj_param == 'profit' else 'expected_revenue'
            
            st.dataframe(opt_df.style.format({
                "base_price": "{:,.0f}", "cost_price": "{:,.0f}", "sim_net_price": "{:,.0f}",
                "pred_qty": "{:,.0f}", "expected_revenue": "{:,.0f}", "expected_profit": "{:,.0f}"
            }).background_gradient(subset=[highlight_col], cmap='Greens'), height=400, use_container_width=True)
            
            # Nút Export
            csv_data = opt_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải xuống File Kế hoạch (CSV)", 
                data=csv_data, 
                file_name=f'Pricing_MasterPlan_{obj_param}.csv', 
                mime='text/csv'
            )

elif page == "5. Giải thích Mô hình (XAI)":
    st.title("🧠 Phân rã Trí tuệ Nhân tạo (Explainable AI)")
    st.markdown("""
    *Màn hình này sử dụng thuật toán **SHAP (SHapley Additive exPlanations)** để minh bạch hóa mô hình hộp đen. 
    Nó giúp chúng ta hiểu chính xác **yếu tố nào** đang thúc đẩy hoặc kéo lùi sức mua của khách hàng.*
    """)
    
    st.divider()
    
    col_x1, col_x2 = st.columns([1, 3])
    with col_x1:
        st.info("""
        **📖 Cách đọc biểu đồ:**
        * **Trục dọc (Y):** Các yếu tố kinh doanh được sắp xếp theo mức độ quan trọng (từ trên xuống dưới).
        * **Màu sắc:** Màu Đỏ (🔴) nghĩa là giá trị của yếu tố đó đang cao, màu Xanh (🔵) là đang thấp.
        * **Trục ngang (X):** Điểm nằm lệch sang **Phải** làm Tăng sức mua. Điểm nằm sang **Trái** làm Giảm sức mua.
        """)
        
        st.success("""
        **💡 Ví dụ Đọc hiểu:**
        Nhìn vào *Mức Khuyến Mãi (%)*, ta thấy các chấm Đỏ (Khuyến mãi cao) đều văng mạnh sang lề phải. 
        Điều này chứng minh máy học đã hiểu đúng quy luật: **Giảm giá càng sâu, khách mua càng nhiều**.
        """)
        
    with col_x2:
        with st.spinner("Đang kết xuất phân rã SHAP..."):
            # Lấy ngẫu nhiên 1500 dòng để vẽ cho nhanh và không bị quá tải chấm
            sample_df = df.sample(n=min(1500, len(df)), random_state=42)
            st.pyplot(charts.plot_shap_summary(model, sample_df, feature_cols))