# 🛍️ Retail Pricing Dashboard (Hệ thống Hỗ trợ Ra Quyết định - DSS)

Đây là một ứng dụng Web App thông minh dựa trên AI/Machine Learning để phân tích dữ liệu bán lẻ, dự báo nhu cầu khách hàng và **tối ưu hóa chiến lược giá & khuyến mãi** cho hệ thống chuỗi cửa hàng. Ứng dụng giúp các nhà quản lý (C-Level/Manager) đưa ra quyết định dựa trên dữ liệu (Data-driven) nhằm tối đa hóa lợi nhuận hoặc doanh thu.

## ✨ Các Tính Năng Chính

Hệ thống cung cấp 5 module phân tích chuyên sâu:

1. **📊 Bảng điều khiển Tổng quan (Overview)**
   * Theo dõi các chỉ số KPI theo thời gian thực (Doanh thu, Sản lượng, Biên lợi nhuận, Mức giảm giá trung bình).
   * Phân tích xu hướng (Trend Analysis) đa cấp độ: Toàn hệ thống, Theo Cửa hàng (Store) và Theo Sản phẩm (SKU).
   * Trực quan hóa tính mùa vụ và top các cửa hàng đóng góp doanh thu.

2. **📈 Phân tích Dữ liệu & Business Insights (Analytics)**
   * **Phát hiện thất thoát (Profit Leakage):** Ước tính số tiền thất thoát do bán dưới giá vốn.
   * **Khoảng giảm giá tối ưu:** Tìm ra mức % giảm giá mang lại lợi nhuận cao nhất.
   * **Phân loại cửa hàng (Store Segmentation):** Tự động phân nhóm thành *Ngôi sao, Đốt tiền khuyến mãi, Tiềm năng tăng trưởng, Hiệu suất thấp* để có hành động can thiệp kịp thời.

3. **⚡ Mô phỏng Kịch bản (What-If Simulation)**
   * Giả lập phản ứng của thị trường (Sức mua & Lợi nhuận) tự động khi người dùng thay đổi mức giảm giá cho 1 sản phẩm cụ thể.
   * AI tự động khuyến nghị mức giá so với hiện tại, cảnh báo nếu quyết định xả kho gây lỗ.

4. **🎯 Đề xuất Tối ưu (Optimization - Master Plan)**
   * Lập bảng kế hoạch giá (Pricing Plan) tự động cho toàn chuỗi dựa trên 2 mục tiêu chiến lược do người quản lý chọn: (1) Tối đa lợi nhuận hoặc (2) Xả kho đẩy doanh thu.
   * Cho phép tải xuống (`.csv`) danh sách kịch bản giá cụ thể để thiết lập lên hệ thống POS/ERP tuần tới.

5. **🧠 Giải thích Mô hình Máy học (Explainable AI - SHAP)**
   * Minh bạch hóa thuật toán AI dạng "hộp đen".
   * Sử dụng biểu đồ SHAP để phân rã mức độ quan trọng và hướng tác động (Tăng/Giảm) của từng yếu tố kinh doanh tới sức mua của khách hàng.

## 🛠️ Công Nghệ Sử Dụng

* **Giao diện & Ứng dụng Web:** Streamlit
* **Trực quan hóa Dữ liệu:** Plotly, Matplotlib, Seaborn
* **Xử lý Dữ liệu:** Pandas, NumPy
* **Mô hình Dự báo (Machine Learning):** LightGBM, Scikit-learn
* **Explainable AI:** SHAP

## 📂 Cấu Trúc Dự Án

```text
retailforecasting_webapp/
│
├── app.py                 # File chạy ứng dụng web chính (Streamlit UI)
├── pipeline.py            # Script dùng để chạy quy trình xử lý dữ liệu và huấn luyện mô hình
├── requirements.txt       # Danh sách các thư viện cần cài đặt
├── README.md              # Tài liệu hướng dẫn (File này)
│
├── data/                  # Nơi chứa các tập dữ liệu đầu vào và đầu ra
│   └── features/          # Dữ liệu phục vụ huấn luyện và chạy mô hình (model_features.csv, ...)
│
├── models/                # Thư mục chứa các tệp mô hình AI đã được huấn luyện (.pkl)
│   └── lightgbm_discount_model.pkl
│
└── src/                   # Thư mục mã nguồn chính (Source)
    ├── models/            # Logic chạy dự báo (predict_sales)
    ├── optimization/      # Logic chạy tối ưu hóa giá & what-if scenario (discount_optimizer)
    └── visualization/     # Các hàm tĩnh tạo biểu đồ (charts.py)
```

## 🚀 Hướng Dẫn Cài Đặt và Chạy Ứng Dụng Khởi Chạy

**Bước 1: Clone hoặc tải mã nguồn về máy local**
Mở terminal/command prompt và trỏ về thư mục dự án:
```bash
cd duong-dan-den-thu-muc/retailforecasting_webapp
```

**Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)**
```bash
python -m venv .venv
# Kích hoạt trên Windows:
.venv\Scripts\activate
# Kích hoạt trên Mac/Linux:
source .venv/bin/activate
```

**Bước 3: Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

**Bước 4: Khởi chạy Ứng dụng Streamlit**
```bash
streamlit run app.py
```
Sau đó, ứng dụng sẽ tự động mở trên trình duyệt mặc định tại vị trí: `http://localhost:8501`.