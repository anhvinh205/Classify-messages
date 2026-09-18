import argparse
import pandas as pd
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_text, create_dictionary, build_feature_matrix
from src.model import encode_labels, split_data, train_model, evaluate_model, save_model

def main(data_path):
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path

    print("Loading dataset ...")
    df = pd.read_csv(data_path)
    print(f"   {len(df)} rows loaded.")

    messages_raw = df["v2"].values.tolist()
    labels_raw   = df["v1"].values.tolist()

    print("Preprocessing text ...")
    messages = [preprocess_text(m) for m in messages_raw]

    print("Encoding labels ...")
    y, le = encode_labels(labels_raw)

    print("Splitting data ...")
    indices = np.arange(len(messages))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_data(indices, y)

    print("Building vocabulary from training data ...")
    dictionary = create_dictionary([messages[i] for i in train_idx])
    print(f"   Vocabulary: {len(dictionary)} tokens")

    print("Creating features ...")
    X_train = build_feature_matrix([messages[i] for i in train_idx], dictionary)
    X_val = build_feature_matrix([messages[i] for i in val_idx], dictionary)
    X_test = build_feature_matrix([messages[i] for i in test_idx], dictionary)

    print("Training ...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_val,  y_val,  "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    save_model(model, dictionary, le)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    args = parser.parse_args()
    main(args.data)