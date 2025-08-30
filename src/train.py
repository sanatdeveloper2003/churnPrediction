import argparse, json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def build_preprocessor(df, target):
    X = df.drop(columns=[target])
    cat = [c for c in X.columns if X[c].dtype == "object"]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat),
    ])
    return pre

def main(data_path, target, outdir, test_size, seed):
    out = Path(outdir); (out/"figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    y = df[target].astype(str); X = df.drop(columns=[target])

    pre = build_preprocessor(df, target)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)

    metrics = {
        "accuracy": float(accuracy_score(yte, ypred)),
        "precision": float(precision_score(yte, ypred, average="weighted")),
        "recall": float(recall_score(yte, ypred, average="weighted")),
        "f1": float(f1_score(yte, ypred, average="weighted")),
    }
    with open(out/"metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    cm = confusion_matrix(yte, ypred, labels=np.unique(y))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest"); ax.set_title("Confusion matrix")
    ax.set_xticks(range(len(np.unique(y)))); ax.set_yticks(range(len(np.unique(y))))
    ax.set_xticklabels(np.unique(y), rotation=45, ha="right"); ax.set_yticklabels(np.unique(y))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax); plt.tight_layout()
    fig.savefig(out/"figures/confusion_matrix.png", dpi=160); plt.close(fig)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model.joblib")
    print("Saved: models/model.joblib, reports/metrics.json, reports/figures/confusion_matrix.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_churn.csv")
    p.add_argument("--target", default="Churn")  # поменяй на название целевой колонки из твоего набора
    p.add_argument("--outdir", default="reports")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.data, args.target, args.outdir, args.test_size, args.seed)
