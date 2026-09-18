# 🛡️ Spam Classifier — Naive Bayes

Phân loại tin nhắn **Spam / Ham** sử dụng thuật toán **Gaussian Naive Bayes**.  
Project thuộc AI Vietnam AIO2024 — Module 02.

---

## 📁 Cấu trúc project

```
spam_classifier/
├── data/
│   └── data.csv                    
├── models/                         
│   ├── naive_bayes.pkl
│   ├── dictionary.pkl
│   └── label_encoder.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            
│   └── model.py                    
├── app.py                          
├── train.py                        
├── requirements.txt
└── README.md
```

---

## ⚡ Hướng dẫn chạy

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Mac/Linux)
source venv/bin/activate
```

### 2. Cài thư viện và tải NLTK resources

Chạy một lệnh duy nhất để cài tất cả:

```bash
pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Huấn luyện model

```bash
cd spam_classifier
python train.py --data data/data.csv
```

Sau khi chạy xong, model sẽ được lưu tự động vào thư mục `models/`.

### 4. Chạy Web App

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

---

## 🧪 Pipeline xử lý

```
Message
  → Lowercase
  → Punctuation Removal
  → Tokenize
  → Remove Stopwords
  → Stemming
  → Bag-of-Words Features
  → Gaussian Naive Bayes
  → Spam / Ham
```

---

## 📊 Kết quả

| Split      | Accuracy |
|------------|----------|
| Validation | ~88%     |
| Test       | ~86%     |

---

## 🛠️ Tech Stack

- `scikit-learn` — Naive Bayes, metrics, preprocessing
- `nltk` — tokenization, stopwords, stemming
- `pandas` / `numpy` — data handling
- `streamlit` — web interface
- `joblib` — model persistence
