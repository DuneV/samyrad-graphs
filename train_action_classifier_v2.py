#!/usr/bin/env python3
"""
train_action_classifier_v2.py
===============================
Entrena un clasificador sobre embeddings CLIP + features de movimiento.
Soporta MLP y HistGradientBoosting (--model), y balanceo de clases
opcional (--balance-mode).

Uso:
    python3 train_action_classifier_v2.py \
        --embeddings embeddings_merged_remapped.npz \
        --output action_classifier_v2.joblib \
        --model hgb
"""

import argparse
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default="action_classifier_v2.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["mlp", "hgb"], default="hgb",
                        help="hgb: HistGradientBoosting (mejor accuracy total "
                             "en tus pruebas). mlp: red neuronal chica.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--min-samples-per-class", type=int, default=6)
    parser.add_argument("--balance-mode", choices=["none", "full", "sqrt"], default="none")
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    X, y_raw = data["X"], data["y"]
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} dims")

    classes, counts = np.unique(y_raw, return_counts=True)
    print("Distribución:", dict(zip(classes, counts)))

    valid = {c for c, n in zip(classes, counts) if n >= args.min_samples_per_class}
    excluded = set(classes) - valid
    if excluded:
        print(f"⚠ Clases excluidas por <{args.min_samples_per_class} muestras: {sorted(excluded)}")
    mask = np.isin(y_raw, list(valid))
    X, y_raw = X[mask], y_raw[mask]

    if len(valid) < 2:
        print("No quedan suficientes clases para entrenar.")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.balance_mode == "none":
        sw = None
    elif args.balance_mode == "full":
        sw = compute_sample_weight("balanced", y_train)
    else:
        sw = np.sqrt(compute_sample_weight("balanced", y_train))

    if args.model == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    else:
        clf = MLPClassifier(hidden_layer_sizes=tuple(args.hidden_layers),
                            max_iter=args.max_iter, early_stopping=True,
                            validation_fraction=0.15, random_state=42)

    if sw is not None:
        clf.fit(X_train_s, y_train, sample_weight=sw)
    else:
        clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)

    print("\n── Reporte de clasificación (set de validación) ──────────")
    print(classification_report(
        le.inverse_transform(y_test), le.inverse_transform(y_pred), zero_division=0))

    labels_sorted = sorted(set(y_raw))
    cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=labels_sorted)
    print("Matriz de confusión (fila=real, columna=predicho):")
    print("          " + " ".join(f"{l[:7]:>8s}" for l in labels_sorted))
    for i, l in enumerate(labels_sorted):
        print(f"{l[:9]:>9s} " + " ".join(f"{cm[i][j]:8d}" for j in range(len(labels_sorted))))

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nModelo: {args.model}  |  Balanceo: {args.balance_mode}")
    print(f"Accuracy global: {acc*100:.1f}%")
    print(f"Balanced accuracy: {bacc*100:.1f}%")

    joblib.dump({"classifier": clf, "scaler": scaler, "label_encoder": le,
                "model_type": args.model}, args.output)
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    main()