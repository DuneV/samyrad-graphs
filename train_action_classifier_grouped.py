#!/usr/bin/env python3
"""
train_action_classifier_grouped.py
=====================================
Versión corregida de train_action_classifier_v2.py: en vez de dividir
train/test por MUESTRA individual (train_test_split con stratify=y),
divide por PARTICIPANTE (StratifiedGroupKFold agrupando por el archivo
de origen — cada embeddings_PXX_v4.npz es un participante).

Esto valida si el 56.8% que obtuvimos con embeddings congelados tenía
el mismo problema de fuga que encontramos en el fine-tuning: ventanas
casi idénticas del mismo segmento (mismo participante, mismo video)
repartidas entre train y test, inflando el accuracy.

Uso:
    python3 train_action_classifier_grouped.py \
        --embeddings-files embeddings_P01_v4.npz embeddings_P02_v4.npz ... \
        --output action_classifier_grouped.joblib
"""

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
import joblib

DEFAULT_REMAP = {
    "pull": "displace", "push": "displace",
    "remove_from": "displace", "inspect": "displace",
    "close": "actuate", "open": "actuate",
}

PARTICIPANT_RE = re.compile(r"(P\d{2})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-files", nargs="+", required=True,
                        help="Archivos .npz POR PARTICIPANTE (no el "
                             "combinado de merge_embeddings.py) — el "
                             "nombre del archivo debe contener el "
                             "participante, ej. embeddings_P01_v4.npz")
    parser.add_argument("--output", default="action_classifier_grouped.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    parser.add_argument("--no-remap", action="store_true")
    args = parser.parse_args()

    remap = {} if args.no_remap else DEFAULT_REMAP

    all_X, all_y, all_groups = [], [], []
    for fpath in args.embeddings_files:
        match = PARTICIPANT_RE.search(Path(fpath).stem)
        participant = match.group(1) if match else Path(fpath).stem
        data = np.load(fpath, allow_pickle=True)
        X, y = data["X"], data["y"]
        y_remapped = np.array([remap.get(lbl, lbl) for lbl in y])
        all_X.append(X)
        all_y.append(y_remapped)
        all_groups.extend([participant] * len(y))
        print(f"{fpath}: {len(y)} muestras del participante {participant}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    classes, counts = np.unique(y, return_counts=True)
    print(f"\nTotal: {len(y)} muestras, {len(set(groups))} participantes")
    print("Distribución:", dict(zip(classes, counts)))

    valid_classes = {c for c, n in zip(classes, counts) if n >= args.min_samples_per_class}
    excluded = set(classes) - valid_classes
    if excluded:
        print(f"⚠ Clases excluidas por <{args.min_samples_per_class} muestras: {sorted(excluded)}")
    mask = np.isin(y, list(valid_classes))
    X, y, groups = X[mask], y[mask], groups[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_splits = max(2, round(1 / args.test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y_enc, groups))

    train_participants = set(groups[train_idx])
    test_participants = set(groups[test_idx])
    overlap = train_participants & test_participants
    assert not overlap, f"Fuga: participantes en ambos splits: {overlap}"
    print(f"\nSplit agrupado por participante: {len(train_participants)} en train "
          f"({sorted(train_participants)}), {len(test_participants)} en test "
          f"({sorted(test_participants)}) — 0 solapados, verificado")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    print("\n── Reporte de clasificación (participantes NUNCA vistos) ──────────")
    print(classification_report(
        le.inverse_transform(y_test), le.inverse_transform(y_pred), zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    print(f"Accuracy global (participantes nunca vistos): {acc*100:.1f}%")
    print(f"Balanced accuracy: {bacc*100:.1f}%")
    print(f"\n(referencia: el split anterior, sin agrupar, había dado 56.8%)")

    joblib.dump({"classifier": clf, "scaler": scaler, "label_encoder": le}, args.output)
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    main()