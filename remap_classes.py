#!/usr/bin/env python3
import argparse
import numpy as np

DEFAULT_MERGE = {
    "pull": "displace",
    "push": "displace",
    "remove_from": "displace",
    "inspect": "displace",
    "close": "actuate",
    "open": "actuate",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default="embeddings_remapped.npz")
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    X, y = data["X"], data["y"]

    classes_before, counts_before = np.unique(y, return_counts=True)
    print("Distribución ANTES:", dict(zip(classes_before, counts_before)))

    y_new = np.array([DEFAULT_MERGE.get(label, label) for label in y])

    classes_after, counts_after = np.unique(y_new, return_counts=True)
    print("Distribución DESPUÉS:", dict(zip(classes_after, counts_after)))
    print(f"\nClases: {len(classes_before)} -> {len(classes_after)}")

    np.savez(args.output, X=X, y=y_new, clip_dim=data.get("clip_dim", X.shape[1] - 4))
    print(f"Guardado en {args.output}")

if __name__ == "__main__":
    main()