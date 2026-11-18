"""
Train the XGBoost model once and save it, so the Streamlit app starts instantly.

Usage:
    python train_model.py                       # uses covtype.data in this folder
    python train_model.py --data path/to/covtype.data
    python train_model.py --fast                # 100 trees, finishes in ~1-2 min
"""
import argparse
import json
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from common import FEATURES, TARGET, load_covtype

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="covtype.data")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()

df = load_covtype(args.data)
X, y = df[FEATURES], df[TARGET] - 1          # XGBoost needs labels 0..6
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=100 if args.fast else 300,
    learning_rate=0.2,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
model.fit(X_tr, y_tr)
train_time = time.time() - t0

pred = model.predict(X_te)
metrics = {
    "accuracy": float(accuracy_score(y_te, pred)),
    "macro_f1": float(f1_score(y_te, pred, average="macro")),
    "train_seconds": round(train_time, 1),
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
    "report": classification_report(y_te, pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_te, pred, normalize="true").tolist(),
    "feature_importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
}

# small stratified sample (~60k rows, ~3 MB) so the deployed app doesn't need the 75 MB file
sample = df.groupby(TARGET, group_keys=False).sample(frac=0.1, random_state=42)
sample.to_csv("covtype_sample.csv.gz", header=False, index=False, compression="gzip")

model.save_model("xgb_covtype.json")
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Done in {train_time:.0f}s | accuracy={metrics['accuracy']:.4f} | macro F1={metrics['macro_f1']:.4f}")
