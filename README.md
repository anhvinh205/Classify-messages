# 🛡️ Spam Classifier

Ứng dụng phân loại tin nhắn `spam` và `ham` bằng Python, scikit-learn và
Streamlit. Model hiện tại sử dụng **balanced Logistic Regression** với
bag-of-words features.

---

## 📁 Cấu trúc project

```text
.
├── app.py
├── train.py
├── requirements.txt
├── data/
│   └── data.csv
├── models/
│   ├── classifier.pkl
│   ├── dictionary.pkl
│   └── label_encoder.pkl
└── src/
    ├── model.py
    └── preprocessing.py
```

---

## ⚡ Cài đặt

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

NLTK resources được tải tự động khi chạy pipeline lần đầu. Nếu môi trường
không có Internet, hãy tải trước:

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 🚀 Huấn luyện

Chạy từ thư mục gốc repository:

```powershell
python train.py
```

Dataset CSV cần có hai cột:

```text
v1: label (ham hoặc spam)
v2: message
```

Pipeline chia dữ liệu theo stratified train/validation/test split và chỉ xây
vocabulary từ training set để tránh data leakage. Model được lưu vào `models/`.

## 🌐 Chạy ứng dụng

```powershell
streamlit run app.py
```

## 📊 Kết quả tham khảo

Đánh giá trên test split của dataset hiện tại:

| Metric | Score |
| --- | ---: |
| Accuracy | 98.21% |
| Spam precision | 100.00% |
| Spam recall | 86.67% |
| Spam F1-score | 92.86% |

## 🧪 Pipeline

```text
Message
  -> lowercase and tokenize
  -> remove stopwords and stem
  -> bag-of-words features
  -> balanced Logistic Regression
  -> Spam / Ham
```

## 🛠️ Tech stack

- `scikit-learn` - model training and metrics
- `nltk` - tokenization, stopwords and stemming
- `pandas` / `numpy` - data processing
- `streamlit` - web interface
- `joblib` - model persistence
