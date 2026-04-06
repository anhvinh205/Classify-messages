
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_text, create_features
from src.model import load_model

# ── Cấu hình trang ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None, None

model, dictionary, le = get_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Spam Detector")
st.markdown("Phân loại tin nhắn **Spam / Ham** sử dụng thuật toán **Naive Bayes**")
st.divider()

# ── Cảnh báo nếu chưa train model ─────────────────────────────────────────────
if model is None:
    st.warning(
        "⚠️ Chưa tìm thấy model!\n\n"
        "Hãy chạy lệnh sau trong terminal:\n"
        "```bash\npython train.py --data data/data.csv\n```"
    )
    st.stop()

# ── Tin nhắn mẫu ───────────────────────────────────────────────────────────────
st.markdown("### 📝 Nhập tin nhắn cần kiểm tra")

col1, col2, col3 = st.columns(3)
selected = None
with col1:
    if st.button("📧 Spam mẫu"):
        selected = "Congratulations! You've won a FREE iPhone. Click here to claim now!"
with col2:
    if st.button("💬 Ham mẫu"):
        selected = "Hey, are you coming to the meeting at 3pm today?"
with col3:
    if st.button("🏦 Scam mẫu"):
        selected = "URGENT: Your bank account suspended. Send OTP to verify immediately."

# ── Ô nhập tin nhắn ────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="Nội dung tin nhắn",
    value=selected or "",
    height=130,
    placeholder="Nhập hoặc dán tin nhắn vào đây...",
)

predict_btn = st.button("🔍 Phân tích", use_container_width=True, type="primary")

# ── Dự đoán ────────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập nội dung tin nhắn!")
        st.stop()

    # Tiền xử lý và dự đoán
    tokens      = preprocess_text(user_input)
    features    = create_features(tokens, dictionary)
    features_2d = np.array(features).reshape(1, -1)

    prediction     = model.predict(features_2d)
    prediction_cls = le.inverse_transform(prediction)[0]
    proba          = model.predict_proba(features_2d)[0]

    st.divider()

    # ── Hiển thị kết quả ───────────────────────────────────────────────────────
    if prediction_cls.lower() == "spam":
        st.error("🚨 **SPAM** — Tin nhắn này có dấu hiệu spam!")
    else:
        st.success("✅ **HAM** — Tin nhắn bình thường!")

    # ── Biểu đồ xác suất ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Xác suất dự đoán")

    fig, ax = plt.subplots(figsize=(6, 2))
    classes = le.classes_
    colors  = ["#2ecc71" if c.lower() == "ham" else "#e74c3c" for c in classes]
    ax.barh(classes, proba * 100, color=colors, height=0.4)

    for i, (p, c) in enumerate(zip(proba, classes)):
        ax.text(p * 100 + 1, i, f"{p*100:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Xác suất (%)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Chi tiết token ─────────────────────────────────────────────────────────
    with st.expander("🔬 Chi tiết tiền xử lý"):
        st.markdown(f"**Số tokens sau preprocessing:** `{len(tokens)}`")
        st.write(tokens)
        st.markdown(f"**Vocab size:** `{len(dictionary)}` từ")