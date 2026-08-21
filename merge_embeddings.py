#!/usr/bin/env python3
"""
merge_embeddings.py
====================
Combina varios archivos .npz de embeddings (uno por participante, ej.
embeddings_P01_v2.npz, embeddings_P02_v2.npz, ...) en un único dataset
para entrenar sobre todos a la vez.

Uso:
    python3 merge_embeddings.py \
        embeddings_P01_v2.npz embeddings_P02_v2.npz embeddings_P03_v2.npz \
        --output embeddings_merged.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Archivos .npz a combinar")
    parser.add_argument("--output", default="embeddings_merged.npz")
    args = parser.parse_args()

    all_X, all_y = [], []
    clip_dim = None

    for path in args.inputs:
        data = np.load(path, allow_pickle=True)
        X, y = data["X"], data["y"]
        this_clip_dim = int(data.get("clip_dim", X.shape[1] - 4))

        if clip_dim is None:
            clip_dim = this_clip_dim
        elif clip_dim != this_clip_dim:
            print(f"⚠ {path} tiene clip_dim={this_clip_dim}, se esperaba "
                  f"{clip_dim} (¿mezclaste ViT-B/32 con ViT-L/14?). Se omite.")
            continue

        all_X.append(X)
        all_y.append(y)
        classes, counts = np.unique(y, return_counts=True)
        print(f"{path}: {len(y)} muestras — {dict(zip(classes, counts))}")

    if not all_X:
        print("Nada para combinar.")
        return

    X_merged = np.concatenate(all_X, axis=0)
    y_merged = np.concatenate(all_y, axis=0)

    print(f"\nTotal combinado: {len(y_merged)} muestras")
    classes, counts = np.unique(y_merged, return_counts=True)
    print("Distribución final:", dict(zip(classes, counts)))

    np.savez(args.output, X=X_merged, y=y_merged, clip_dim=clip_dim)
    print(f"\nGuardado en {args.output}")


if __name__ == "__main__":
    main()