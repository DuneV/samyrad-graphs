#!/usr/bin/env python3
"""
compare_classifiers.py
========================
Prueba varios clasificadores sobre el MISMO split train/test de tus
embeddings reales, para decidir con evidencia real (no simulaciones)
cuál conviene usar.

Uso:
    python3 compare_classifiers.py --embeddings embeddings_merged_remapped.npz
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--min-samples-per-class", type=int, default=6)
    parser.add_argument("--balance-mode", choices=["none", "full"], default="none")
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    X, y_raw = data["X"], data["y"]
    classes, counts = np.unique(y_raw, return_counts=True)
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} dims, "
          f"clases: {dict(zip(classes, counts))}\n")

    valid = {c for c, n in zip(classes, counts) if n >= args.min_samples_per_class}
    mask = np.isin(y_raw, list(valid))
    X, y_raw = X[mask], y_raw[mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sw = compute_sample_weight("balanced", y_train) if args.balance_mode == "full" else None

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight=("balanced" if args.balance_mode == "full" else None)),
        "MLP (256,64)": MLPClassifier(
            hidden_layer_sizes=(256, 64), max_iter=500, early_stopping=True, random_state=42),
        "MLP (512,128,32)": MLPClassifier(
            hidden_layer_sizes=(512, 128, 32), max_iter=500, early_stopping=True, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight=("balanced" if args.balance_mode == "full" else None)),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight=("balanced" if args.balance_mode == "full" else None)),
    }

    print(f"{'Modelo':22s} {'Accuracy':>10s} {'Balanced':>10s} {'Tiempo':>8s}")
    print("-" * 55)
    results = []
    for name, model in models.items():
        t0 = time.time()
        try:
            if sw is not None and "sample_weight" in model.fit.__code__.co_varnames:
                model.fit(X_train_s, y_train, sample_weight=sw)
            else:
                model.fit(X_train_s, y_train)
        except TypeError:
            model.fit(X_train_s, y_train)   # modelos que no aceptan sample_weight
        elapsed = time.time() - t0

        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        results.append((name, acc, bacc))
        print(f"{name:22s} {acc*100:9.1f}% {bacc*100:9.1f}% {elapsed:7.1f}s")

    best = max(results, key=lambda r: r[1])
    print(f"\nMejor accuracy total: {best[0]} ({best[1]*100:.1f}%)")


if __name__ == "__main__":
    main()