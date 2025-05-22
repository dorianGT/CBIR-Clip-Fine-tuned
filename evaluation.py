import os
import json
import numpy as np
from faiss_index import load_embeddings, load_index_faiss, search
from parse_groundtruth import load_groundtruth_json
from projection_and_metrics import (save_projection, save_histogram_distance, save_colored_projection,
                                    recall_at_k,precision_at_k,mean_average_precision,f1_score_at_k,
                                    evaluate_with_distance_threshold)

def evaluate(model_folder):
    """
    Évalue les performances du modèle en utilisant plusieurs métriques et visualisations.

    Cette fonction :
    - Charge les embeddings, l'index FAISS et les groupes ground-truth.
    - Génère des visualisations (projections t-SNE/UMAP, projections colorées, histogramme de distances).
    - Calcule les métriques suivantes :
        - Recall@1, @5, @10
        - Precision@1, @5, @10
        - mean Average Precision (mAP@10)
        - F1-score@5
        - Seuil optimal basé sur la distance moyenne pour F1-score, précision et rappel

    Les résultats sont sauvegardés dans `model_folder/evaluation/metrics.json`
    et les visualisations sont également sauvegardées dans ce dossier.

    Args:
        model_folder (str): Chemin vers le dossier contenant les fichiers du modèle, embeddings et index.

    Returns:
        dict: Un dictionnaire contenant toutes les métriques calculées, avec les clés :
              "recall@1", "recall@5", "recall@10",
              "precision@1", "precision@5", "precision@10",
              "mAP@10", "f1@5",
              "threshold", "precision_thresh", "recall_thresh", "f1_thresh"
    """
    os.makedirs(os.path.join(model_folder, "evaluation"), exist_ok=True)
    
    # Chargement
    embeddings, paths = load_embeddings(os.path.join(model_folder, "embeddings.npz"))
    index = load_index_faiss(os.path.join(model_folder, "faiss.index"))
    groups, _, _, _, _  = load_groundtruth_json("ground_truth.json")

    # Projections visuelles
    save_projection(embeddings, os.path.join(model_folder, "evaluation/tsne_projection.png"), method="tsne")
    save_projection(embeddings, os.path.join(model_folder, "evaluation/umap_projection.png"), method="umap")

    save_colored_projection(embeddings, paths, groups, os.path.join(model_folder, "evaluation/tsne_colored.png"), method="tsne")
    save_colored_projection(embeddings, paths, groups, os.path.join(model_folder, "evaluation/umap_colored.png"), method="umap")

    # Distances histogram
    D, _ = index.search(embeddings.astype(np.float32), 6)
    save_histogram_distance(D, os.path.join(model_folder, "evaluation/distance_histogram.png"))

    # Évaluation Recall@K
    recall_k1 = recall_at_k(index, embeddings, paths, groups, k=1)
    recall_k5 = recall_at_k(index, embeddings, paths, groups, k=5)
    recall_k10 = recall_at_k(index, embeddings, paths, groups, k=10)

    # Évaluation Precision@K
    precision_k1 = precision_at_k(index, embeddings, paths, groups, k=1)
    precision_k5 = precision_at_k(index, embeddings, paths, groups, k=5)
    precision_k10 = precision_at_k(index, embeddings, paths, groups, k=10)

    # Évaluation mAP@5 mAP@10
    map_k5 = mean_average_precision(index, embeddings, paths, groups, k=5)
    map_k10 = mean_average_precision(index, embeddings, paths, groups, k=10)

    # Évaluation f1@5 f1@10
    f1_k5 = f1_score_at_k(index, embeddings, paths, groups, k=5)
    f1_k10 = f1_score_at_k(index, embeddings, paths, groups, k=10)

    # Évaluation basée sur un seuil de distance
    threshold, precision, recall, f1 = evaluate_with_distance_threshold(index, embeddings, paths, groups, k=10, threshold_type="mean")

    # Résultats
    results = {
        "recall@1": round(recall_k1, 4),
        "recall@5": round(recall_k5, 4),
        "recall@10": round(recall_k10, 4),
        "precision@1": round(precision_k1, 4),
        "precision@5": round(precision_k5, 4),
        "precision@10": round(precision_k10, 4),
        "mAP@5": round(map_k5, 4),
        "mAP@10": round(map_k10, 4),
        "f1@5": round(f1_k5, 4),
        "f1@10": round(f1_k10, 4),
        "threshold": round(threshold, 4),
        "precision_thresh": round(precision, 4),
        "recall_thresh": round(recall, 4),
        "f1_thresh": round(f1, 4)
    }

    with open(os.path.join(model_folder, "evaluation/metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Évaluation terminée et résultats sauvegardés.")
    print(json.dumps(results, indent=4))

    return results