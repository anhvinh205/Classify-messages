
import joblib
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

TEST_SIZE = 0.10        # 10% for test
VAL_RATIO = 0.2222     # 20% of total ≈ 20/90 of remaining after test split
SEED      = 42
MODEL_DIR = Path(__file__).parent.parent / "models"


def split_data(X: np.ndarray, y: np.ndarray):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_RATIO, shuffle=True, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """Train a Gaussian Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: GaussianNB, X, y, split_name: str = "Test") -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    cm     = confusion_matrix(y, y_pred)
    print(f"\n{'='*40}")
    print(f"  {split_name} Accuracy: {acc:.4f}")
    print(f"{'='*40}")
    print(classification_report(y, y_pred, target_names=["Ham", "Spam"]))
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "y_pred": y_pred}


def encode_labels(labels: list):
    """Encode string labels → integers. Returns (encoded_y, fitted_encoder)."""
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return y, le


def save_model(model, dictionary: list, le: LabelEncoder):
    """Persist model artifacts to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model,      MODEL_DIR / "naive_bayes.pkl")
    joblib.dump(dictionary, MODEL_DIR / "dictionary.pkl")
    joblib.dump(le,         MODEL_DIR / "label_encoder.pkl")
    print(f"✅  Model saved to {MODEL_DIR}")


def load_model():
    """Load saved model artifacts. Returns (model, dictionary, label_encoder)."""
    model      = joblib.load(MODEL_DIR / "naive_bayes.pkl")
    dictionary = joblib.load(MODEL_DIR / "dictionary.pkl")
    le         = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return model, dictionary, le