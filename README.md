# Shopping Cart Analysis

Phân tích dữ liệu bán lẻ để tìm ra mối quan hệ giữa các sản phẩm thường được mua cùng nhau bằng các kỹ thuật  
**Association Rule Mining (Apriori)**.  

Project triển khai pipeline đầy đủ từ xử lý dữ liệu → phân tích → khai thác luật → sinh báo cáo.

---

## 📌 Thông tin bài tập nhóm

- **Tên nhóm:** Nhom2-cntt  
- **Hình thức:** Bài tập nhóm – Quản lý mã nguồn với GitHub  
- **Repository nhóm:** https://github.com/Nhom2-cntt/shopping_cart_analysis  

### 👥 Thành viên nhóm

| STT | Họ và tên |
|----|----------|--------|
| 1 | Nguyễn Minh Tuấn | 
| 2 | Hoàng Văn Xanh|
| 3 | Phan Tự Nghiệp |
| 3 | Trần Quang Huy |

### 🔄 Quy trình làm việc với Git

1. Fork repository từ giảng viên vào GitHub Organization của nhóm  
2. Clone repository nhóm về máy cá nhân  
3. Mỗi thành viên làm việc trên **branch riêng**  
4. Commit code theo từng chức năng  
5. Push branch lên GitHub  
6. Tạo Pull Request để merge vào `main`  
7. Luôn pull `main` trước khi thực hiện nhiệm vụ mới  

⚠️ **Không chỉnh sửa code trực tiếp trên nhánh `main`.**

---

## ✨ Features

- Làm sạch dữ liệu & xử lý giá trị lỗi  
- Xây dựng basket matrix (transaction × product)  
- Khai phá tập mục phổ biến (Frequent Itemsets)  
- Sinh luật kết hợp (Association Rules)  
- Các chỉ số:
  - Support  
  - Confidence  
  - Lift  
- Visualization:
  - Bar chart  
  - Scatter plot  
  - Network graph  
  - Biểu đồ tương tác Plotly  
- Tự động hóa pipeline bằng **Papermill**

---

## 📂 Project Structure

```text
shopping_cart_analysis/
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   └── processed/
│       ├── cleaned_uk_data.csv
│       ├── basket_bool.parquet
│       └── rules_apriori_filtered.csv
│
├── notebooks/
│   ├── preprocessing_and_eda.ipynb
│   ├── basket_preparation.ipynb
│   ├── apriori_modelling.ipynb
│   └── runs/
│       ├── preprocessing_and_eda_run.ipynb
│       ├── basket_preparation_run.ipynb
│       └── apriori_modelling_run.ipynb
│
├── src/
│   └── apriori_library.py
│
├── run_papermill.py
├── requirements.txt
└── README.md
