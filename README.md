# Stroke Risk EDA

## Cấu trúc thư mục

```text
data/raw/       Dữ liệu gốc (không chỉnh sửa)
docs/           Tài liệu mô tả dự án
notebooks/      Notebook EDA có giải thích từng bước
src/            Mã Python có thể chạy lại
outputs/eda/    Biểu đồ và bảng kết quả EDA
```

## Chạy EDA

Mở `notebooks/01_eda_stroke.ipynb` bằng Jupyter Notebook hoặc VS Code và chạy lần lượt các cell.

Notebook cũng có phần so sánh `class_weight`, SMOTE, ADASYN và threshold tuning. Metric chính là AUC-PR; Recall và F1 được dùng làm metric phụ.

Cài thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Có thể chạy tự động bằng lệnh:

```bash
python src/eda_stroke.py
```
