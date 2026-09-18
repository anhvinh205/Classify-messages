# Spam Classifier

Ứng dụng phân loại tin nhắn `spam` và `ham` bằng Python, scikit-learn và Streamlit.
Mô hình sử dụng Logistic Regression với cân bằng class để ưu tiên chất lượng
nhận diện Spam.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

## Huấn luyện model

Chạy từ thư mục `spam_classifier`:

```powershell
python train.py
```

Hoặc chỉ định dataset khác:

```powershell
python train.py --data data\data.csv
```

Pipeline chia dữ liệu theo tỷ lệ train/validation/test, xây vocabulary chỉ từ
training set để tránh data leakage, sau đó lưu model vào thư mục `models`.

Kết quả test tham khảo trên dataset hiện tại:

```text
Accuracy:  98.21%
Spam precision: 100.00%
Spam recall:    86.67%
Spam F1-score:  92.86%
```

Dataset CSV cần có hai cột:

```text
v1: label (ham hoặc spam)
v2: message
```

## Chạy ứng dụng

```powershell
streamlit run app.py
```

## Cấu trúc chính

```text
spam_classifier/
├── app.py
├── train.py
├── data/
├── models/
└── src/
    ├── model.py
    └── preprocessing.py
```
