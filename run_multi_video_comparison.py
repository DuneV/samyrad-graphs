#!/usr/bin/env python3
"""
run_multi_video_comparison.py
================================
Corre STEP 2 + STEP 5 (clasificar con el modelo entrenado Y con ground
truth, luego inferencia del GNN sobre ambos) para VARIOS videos, y
genera gráficas estilo paper (PDF, listas para \\includegraphics) con
los resultados agregados.

Uso:
    python3 run_multi_video_comparison.py \
        --test-videos kitchen/EPIC-KITCHENS/P37/videos/P37_101.MP4 \
                      kitchen/EPIC-KITCHENS/P37/videos/P37_102.MP4 \
                      kitchen/EPIC-KITCHENS/P37/videos/P37_103.MP4 \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --classifier-checkpoint action_classifier_lora.pth \
        --gnn-checkpoint gnn_cost_optimizer.pth \
        --knowledge-base-path knowledge_base.json \
        --robot-actions-path robot_actions.json
"""

import argparse
import cv2
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


def accuracy_vs_ground_truth(predicted_actions, gt_actions):
    if not gt_actions:
        return float("nan")
    pred_counts = Counter(predicted_actions)
    gt_counts = Counter(gt_actions)
    correct = sum(min(pred_counts[a], gt_counts[a]) for a in gt_counts)
    return correct / sum(gt_counts.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-videos", nargs="+", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--knowledge-base-path", default="knowledge_base.json")
    parser.add_argument("--robot-actions-path", default="robot_actions.json")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output-json", default="multi_video_results.json")
    parser.add_argument("--output-plots-dir", default="paper_figures")
    parser.add_argument("--save-failures-rate", type=float, default=0.10,
                        help="Fracción de las ventanas MAL clasificadas "
                             "que se guardan a disco para revisión manual "
                             "(0.10 = 10%%, no todas -- para no duplicar "
                             "la base de datos)")
    parser.add_argument("--failures-dir", default="failure_samples")
    parser.add_argument("--failures-seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    from epic_ground_truth import EpicGroundTruth
    from trained_classifier_inference import TrainedActionClassifier
    from pipeline_step2_using import build_scenarios_dual
    from neopath.semantic_knowledge import KnowledgeBase
    from neopath.robot_physical_capacities import RobotCapabilities
    from neopath.semantic_graph import SemanticActionGraph
    from neopath.gnn import GNNCostOptimizer
    from benchmark_comparison import build_perceptual_graph, run_gnn_inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = EpicGroundTruth(args.gt_csv)
    classifier = TrainedActionClassifier.load(args.classifier_checkpoint, device=device)
    yolo = YOLO(args.model_path).to(device)
    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    hands = HandLandmarker.create_from_options(hand_options)

    kb = KnowledgeBase(storage_path=args.knowledge_base_path)
    rc = RobotCapabilities(actions_file=args.robot_actions_path)
    sg = SemanticActionGraph(rc, kb)
    sg.build_full_action_graph(kb.concepts.keys(), rc.actions)
    gnn = GNNCostOptimizer(kb, pretrained_path=args.gnn_checkpoint)

    video_id_re = re.compile(r"(P\d{2}_\d{2,3})")
    results = []

    rng = random.Random(args.failures_seed)
    Path(args.failures_dir).mkdir(exist_ok=True)
    failure_log = []
    n_failures_seen, n_failures_saved = 0, 0

    def on_window(crops, pred_action, pred_conf, gt_action, obj, video_id, window_idx):
        nonlocal n_failures_seen, n_failures_saved
        if pred_action == gt_action:
            return   # solo nos interesan los fallos
        n_failures_seen += 1
        if rng.random() >= args.save_failures_rate:
            return   # este fallo no entra en la muestra

        dest_dir = Path(args.failures_dir) / video_id / f"window_{window_idx:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crops):
            cv2.imwrite(str(dest_dir / f"f{i}.jpg"), crop)
        (dest_dir / "info.txt").write_text(
            f"video_id: {video_id}\nobjeto: {obj}\n"
            f"real: {gt_action}\npredicho: {pred_action}\n"
            f"confianza: {pred_conf:.3f}\n")
        failure_log.append({
            "video_id": video_id, "window_idx": window_idx, "objeto": obj,
            "real": gt_action, "predicho": pred_action, "confianza": pred_conf,
        })
        n_failures_saved += 1

    for video_path in args.test_videos:
        video_id_match = video_id_re.search(Path(video_path).stem)
        video_id = video_id_match.group(1) if video_id_match else Path(video_path).stem
        print(f"\n{'='*60}\n{video_id}\n{'='*60}")

        if not gt.has_video(video_id):
            print(f"  ⚠ sin ground truth, se salta")
            continue

        scenario_classifier, scenario_ground_truth = build_scenarios_dual(
            video_path, video_id, gt, classifier, yolo, hands, args.confidence,
            on_window=on_window)

        new_concepts = set(scenario_classifier.get("detected_objects", [])) - set(kb.concepts.keys())
        if new_concepts:
            for c in new_concepts:
                kb.learn_concept(c)
            sg.build_full_action_graph(new_concepts, rc.actions)

        perceptual_graph_a = build_perceptual_graph(scenario_classifier["object_positions"])
        gnn.initialize_model(sg.graph, perceptual_graph_a, scenario_classifier["goal"])
        time_gnn_a, _ = run_gnn_inference(gnn, sg, scenario_classifier)
        time_gnn_b, _ = run_gnn_inference(gnn, sg, scenario_ground_truth)

        pred_actions = [e["action"] for e in scenario_classifier["semantic_line"]]
        gt_actions = [e["action"] for e in scenario_ground_truth["semantic_line"]]
        acc = accuracy_vs_ground_truth(pred_actions, gt_actions)

        result = {
            "video_id": video_id, "n_events": len(gt_actions),
            "accuracy": acc, "time_gnn_classifier": time_gnn_a,
            "time_gnn_oracle": time_gnn_b,
        }
        results.append(result)
        print(f"  accuracy={acc*100:.1f}%  n_eventos={len(gt_actions)}  "
              f"tiempo_gnn={time_gnn_a*1000:.1f}ms")

    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"\nResultados guardados en {args.output_json}")

    if failure_log:
        failure_log_path = Path(args.failures_dir) / "failure_log.json"
        failure_log_path.write_text(json.dumps(failure_log, indent=2))
    pct_saved = (n_failures_saved / n_failures_seen * 100) if n_failures_seen else 0
    print(f"\nFallos totales encontrados: {n_failures_seen}")
    print(f"Fallos guardados a disco ({args.save_failures_rate*100:.0f}% "
          f"objetivo, {pct_saved:.1f}% real): {n_failures_saved}")
    print(f"Imágenes en: {args.failures_dir}/<video_id>/window_XXXXX/")

    if not results:
        print("Sin resultados para graficar.")
        return

    make_plots(results, args.output_plots_dir)


# ── Paleta y estilo tipo DeepMind: sans-serif, colores planos y
# saturados, sin bordes/ejes de más, gridlines horizontales sutiles ──
_DM_BLUE = "#3B82C4"
_DM_CORAL = "#EE6C4D"
_DM_GREEN = "#4C9A6E"
_DM_GRAY = "#7A7A7A"
_DM_GRID = "#E5E5E5"


def _apply_deepmind_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": _DM_GRAY,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "ytick.left": False,
        "text.color": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": _DM_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })


def make_plots(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(exist_ok=True)
    _apply_deepmind_style()

    video_ids = [r["video_id"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    mean_acc = float(np.mean(accuracies))

    # ── Gráfica 1: accuracy por video ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(video_ids, accuracies, color=_DM_BLUE, width=0.55,
                  zorder=3, edgecolor="none")
    ax.axhline(mean_acc, color=_DM_CORAL, linestyle="--", linewidth=1.6,
              zorder=4, label=f"Media: {mean_acc:.1f}%")
    ax.set_title("Accuracy de NEOPATH vs. ground truth", loc="left", pad=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Video (participante nunca visto en entrenamiento)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper right")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 2.5, f"{acc:.1f}%",
                ha="center", fontsize=10.5, color="#333333", fontweight="medium")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_por_video.pdf")
    plt.savefig(f"{output_dir}/accuracy_por_video.png", dpi=300)
    plt.close()

    # ── Gráfica 2: tiempo de inferencia GNN, clasificador vs oracle ──
    times_a = [r["time_gnn_classifier"] * 1000 for r in results]
    times_b = [r["time_gnn_oracle"] * 1000 for r in results]
    x = np.arange(len(video_ids))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width/2, times_a, width, label="Grafo (clasificador)",
          color=_DM_BLUE, zorder=3, edgecolor="none")
    ax.bar(x + width/2, times_b, width, label="Grafo (ground truth / oracle)",
          color=_DM_GREEN, zorder=3, edgecolor="none")
    ax.set_title("Tiempo de inferencia del GNN", loc="left", pad=14)
    ax.set_ylabel("Tiempo (ms)")
    ax.set_xlabel("Video")
    ax.set_xticks(x)
    ax.set_xticklabels(video_ids)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tiempo_inferencia.pdf")
    plt.savefig(f"{output_dir}/tiempo_inferencia.png", dpi=300)
    plt.close()

    print(f"\nGráficas guardadas en {output_dir}/:")
    print(f"  accuracy_por_video.pdf / .png")
    print(f"  tiempo_inferencia.pdf / .png")
    print(f"\nAccuracy promedio across {len(results)} videos: {mean_acc:.1f}%")

    make_latex_table(results, output_dir)


def make_latex_table(results, output_dir):
    """Genera una tabla LaTeX (estilo booktabs, igual que el resto del
    paper) con los datos de esta corrida, lista para \\input{} o copiar
    directo a la sección de resultados."""
    video_ids = [r["video_id"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    n_events = [r["n_events"] for r in results]
    times_a = [r["time_gnn_classifier"] * 1000 for r in results]
    times_b = [r["time_gnn_oracle"] * 1000 for r in results]
    mean_acc = float(np.mean(accuracies))

    rows = "\n".join(
        f"    {vid.replace('_', '\\_')} & {n} & {acc:.1f}\\% & {ta:.1f} & {tb:.1f} \\\\"
        for vid, n, acc, ta, tb in zip(video_ids, n_events, accuracies, times_a, times_b)
    )

    table = f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{NEOPATH evaluado sobre {len(results)} videos egocéntricos
(participantes nunca vistos en entrenamiento). Accuracy \\%
$=$ predicción del clasificador vs.\\ verbo real de EPIC-KITCHENS.}}
\\label{{tab:multi_video_results}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Video}} & \\textbf{{N eventos}} & \\textbf{{Accuracy}} &
\\textbf{{$t_{{\\text{{GNN}}}}$ clf.\\ (ms)}} & \\textbf{{$t_{{\\text{{GNN}}}}$ oracle (ms)}} \\\\
\\midrule
{rows}
\\midrule
\\textbf{{Media}} & -- & \\textbf{{{mean_acc:.1f}\\%}} &
{np.mean(times_a):.1f} & {np.mean(times_b):.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    table_path = Path(output_dir) / "results_table.tex"
    table_path.write_text(table)
    print(f"  results_table.tex (para pegar en el paper)")


if __name__ == "__main__":
    main()