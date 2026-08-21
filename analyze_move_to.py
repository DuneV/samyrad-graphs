#!/usr/bin/env python3
"""
analyze_move_to.py
====================
Analiza la clase "move_to" (la más heterogénea y la que domina la matriz
de confusión) usando el verbo original de EPIC-KITCHENS guardado en los
.npz de v4/xclip (campo "epic_verb").

Dos análisis:
1. Distribución de verbos dentro de move_to — cuáles son más frecuentes.
2. Clustering (KMeans) sobre los embeddings REALES de las muestras de
   move_to, para ver si el modelo mismo "ve" subgrupos naturales — más
   confiable que agrupar verbos a mano por similitud semántica, porque
   se basa en cómo se ven realmente en video, no en cómo suenan sus nombres.

Uso:
    python3 analyze_move_to.py --embeddings embeddings_P01_v4.npz --n-clusters 4
"""

import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--target-class", default="move_to")
    parser.add_argument("--n-clusters", type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    if "epic_verb" not in data:
        print("Este .npz no tiene el campo 'epic_verb' — necesitas uno "
              "extraído con extract_training_embeddings_v4.py o "
              "extract_training_embeddings_xclip.py (v2/v3 no lo guardan).")
        return

    X, y, verbs = data["X"], data["y"], data["epic_verb"]

    mask = y == args.target_class
    n_total = mask.sum()
    print(f"'{args.target_class}': {n_total} muestras totales\n")

    print("── 1. Distribución de verbos EPIC-KITCHENS dentro de "
          f"'{args.target_class}' ──────────")
    verb_counts = Counter(verbs[mask])
    for verb, count in verb_counts.most_common():
        pct = count / n_total * 100
        bar = "█" * int(pct / 2)
        print(f"  {verb:15s} {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n── 2. Clustering (k={args.n_clusters}) sobre los embeddings "
          f"reales ──────────")
    X_target = X[mask]
    # Excluye las últimas 4 dims (features de movimiento) del clustering
    # visual — nos interesa agrupar por CÓMO SE VE la acción
    clip_dim = int(data.get("clip_dim", X.shape[1] - 4))
    X_visual = X_target[:, :clip_dim]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_visual)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    print(f"\nComposición de verbos por cluster (qué tan 'puro' es cada uno):")
    for c in range(args.n_clusters):
        cluster_mask = cluster_labels == c
        cluster_verbs = verbs[mask][cluster_mask]
        n_cluster = len(cluster_verbs)
        verb_dist = Counter(cluster_verbs).most_common(5)
        print(f"\n  Cluster {c} ({n_cluster} muestras, "
              f"{n_cluster/n_total*100:.0f}% del total):")
        for verb, count in verb_dist:
            print(f"    {verb:15s} {count:4d} ({count/n_cluster*100:5.1f}% del cluster)")

    print(f"\n{'='*60}")
    print("Interpretación:")
    print("- Si un cluster tiene un verbo dominante (>60-70%), ese cluster")
    print("  probablemente corresponde a una acción visualmente coherente")
    print("  que vale la pena separar como su propia clase.")
    print("- Si los clusters mezclan verbos parejo, la heterogeneidad de")
    print("  move_to no se explica por el verbo (puede ser ángulo de cámara,")
    print("  objeto, mano izq/der, etc.) y dividir por verbo no ayudaría.")


if __name__ == "__main__":
    main()