import os
import json
import numpy as np
import time
from faiss_index import load_embeddings, load_index_faiss, search
from parse_groundtruth import load_groundtruth_json
from projection_and_metrics import (save_projection, save_histogram_distance, save_colored_projection,
                                    recall_at_k,precision_at_k,mean_average_precision,f1_score_at_k,
                                    evaluate_with_distance_threshold,evaluate_with_distance_no_faiss,
                                    recall_at_k_no_faiss,precision_at_k_no_faiss,mean_average_precision_no_faiss,f1_score_at_k_no_faiss,)

def get_groups_image_only(groups):
    """
    Extrait uniquement les chemins d'images des groupes.

    Args:
        groups (list): Liste de groupes, chaque groupe étant une liste de dictionnaires contenant au moins une clé "image".

    Returns:
        list: Liste de groupes d'images (liste de listes de chemins d'images).
    """
    image_groups = []
    for group in groups:
        images = [item["image"] for item in group]
        image_groups.append(images)
    return image_groups

def eval_with_faiss(model_folder,embeddings_name,faiss_name,groups,label):
    """
    Évalue les performances avec ou sans FAISS sur un ensemble d'embeddings.

    Cette fonction :
    - Charge les embeddings et l'index FAISS.
    - Génère des visualisations (projections t-SNE/UMAP, projections colorées, histogramme des distances).
    - Calcule les métriques suivantes pour FAISS et une recherche brute (sans FAISS) :
        - Recall@1, @5, @10
        - Precision@1, @5, @10
        - mean Average Precision (mAP@5, @10)
        - F1-score@5, @10
        - Évaluation à seuil basé sur la distance moyenne
    - Mesure le temps nécessaire pour calculer toutes les métriques (FAISS et sans FAISS).

    Args:
        model_folder (str): Chemin vers le dossier contenant les fichiers du modèle.
        embeddings_name (str): Nom du fichier .npz contenant les embeddings.
        faiss_name (str): Nom du fichier FAISS contenant l'index.
        groups (list): Liste de groupes d'images pour le ground-truth.
        label (str): Label utilisé pour nommer les fichiers de sortie (ex: "image", "text", "combined").

    Returns:
        dict: Résultats d'évaluation organisés en deux sous-dictionnaires "faiss" et "no_faiss", avec les clés suivantes :
            - "recall@1", "recall@5", "recall@10"
            - "precision@1", "precision@5", "precision@10"
            - "mAP@5", "mAP@10"
            - "f1@5", "f1@10"
            - "threshold" (seulement pour FAISS) : seuil basé sur la distance moyenne
            - "precision_thresh", "recall_thresh", "f1_thresh"
            - "time" : temps en secondes pour le calcul de toutes les métriques de chaque méthode
    """
    # Chargement
    embeddings, paths = load_embeddings(os.path.join(model_folder, embeddings_name))
    index = load_index_faiss(os.path.join(model_folder, faiss_name))

    # Projections visuelles
    save_projection(embeddings, os.path.join(model_folder, f"evaluation/tsne_projection_{label}.png"), method="tsne")
    save_projection(embeddings, os.path.join(model_folder, f"evaluation/umap_projection_{label}.png"), method="umap")

    save_colored_projection(embeddings, paths, groups, os.path.join(model_folder, f"evaluation/tsne_colored_{label}.png"), method="tsne")
    save_colored_projection(embeddings, paths, groups, os.path.join(model_folder, f"evaluation/umap_colored_{label}.png"), method="umap")

    ## Evaluation avec Faiss
    start_faiss = time.time()

    # Distances histogram
    D, _ = index.search(embeddings.astype(np.float32), 6)
    save_histogram_distance(D, os.path.join(model_folder, f"evaluation/distance_histogram_{label}.png"))

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

    faiss_time = time.time() - start_faiss



    ### ÉVALUATION SANS FAISS ###
    start_no_faiss = time.time()

    # Évaluation Recall@K
    recall_k1_no_faiss = recall_at_k_no_faiss(embeddings, paths, groups, k=1)
    recall_k5_no_faiss = recall_at_k_no_faiss(embeddings, paths, groups, k=5)
    recall_k10_no_faiss = recall_at_k_no_faiss(embeddings, paths, groups, k=10)

    # Évaluation Precision@K
    precision_k1_no_faiss = precision_at_k_no_faiss(embeddings, paths, groups, k=1)
    precision_k5_no_faiss = precision_at_k_no_faiss(embeddings, paths, groups, k=5)
    precision_k10_no_faiss = precision_at_k_no_faiss(embeddings, paths, groups, k=10)

    # Évaluation mAP@5 mAP@10
    map_k5_no_faiss = mean_average_precision_no_faiss(embeddings, paths, groups, k=5)
    map_k10_no_faiss = mean_average_precision_no_faiss(embeddings, paths, groups, k=10)

    # Évaluation f1@5 f1@10
    f1_k5_no_faiss = f1_score_at_k_no_faiss(embeddings, paths, groups, k=5)
    f1_k10_no_faiss = f1_score_at_k_no_faiss(embeddings, paths, groups, k=10)

    # Évaluation basée sur un seuil de distance
    threshold_no_faiss, precision_no_faiss, recall_no_faiss, f1_no_faiss = evaluate_with_distance_no_faiss(embeddings, paths, groups, k=10, threshold_type="mean")

    no_faiss_time = time.time() - start_no_faiss

    # Résultats
    results = {
        "faiss": {
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
            "f1_thresh": round(f1, 4),
            "time_sec": round(faiss_time, 2)
        },
        "no_faiss": {
            "recall@1": round(recall_k1_no_faiss, 4),
            "recall@5": round(recall_k5_no_faiss, 4),
            "recall@10": round(recall_k10_no_faiss, 4),
            "precision@1": round(precision_k1_no_faiss, 4),
            "precision@5": round(precision_k5_no_faiss, 4),
            "precision@10": round(precision_k10_no_faiss, 4),
            "mAP@5": round(map_k5_no_faiss, 4),
            "mAP@10": round(map_k10_no_faiss, 4),
            "f1@5": round(f1_k5_no_faiss, 4),
            "f1@10": round(f1_k10_no_faiss, 4),
            "threshold": round(threshold_no_faiss, 4),
            "precision_thresh": round(precision_no_faiss, 4),
            "recall_thresh": round(recall_no_faiss, 4),
            "f1_thresh": round(f1_no_faiss, 4),
            "time_sec": round(no_faiss_time, 2)
        }
    }

    return results

def evaluate(model_folder):
    """
    Évalue globalement les performances du modèle (image, texte, combinaison).

    Cette fonction :
    - Charge les ground-truths.
    - Effectue les évaluations individuelles pour les embeddings image, texte et combinés.
    - Sauvegarde les résultats dans un fichier JSON.
    - Génère toutes les visualisations associées.

    Args:
        model_folder (str): Chemin vers le dossier contenant les fichiers embeddings, index FAISS et le sous-dossier `evaluation/`.

    Returns:
        dict: Résultats complets contenant trois sous-ensembles :
            - "results_image": Résultats pour embeddings image.
            - "results_text": Résultats pour embeddings texte.
            - "results_combined": Résultats pour embeddings combinés.
    """
    os.makedirs(os.path.join(model_folder, "evaluation"), exist_ok=True)
    
    groups, _, _, _, _  = load_groundtruth_json("ground_truth.json")
    groups = get_groups_image_only(groups)

    results_image = eval_with_faiss(model_folder,"embeddings_image.npz","faiss_image.index",groups,"image")
    results_text = eval_with_faiss(model_folder,"embeddings_text.npz","faiss_text.index",groups,"text")
    results_combined = eval_with_faiss(model_folder,"embeddings_combined.npz","faiss_combined.index",groups,"combined")

    # Résultats
    results = {
        "results_image": results_image,
        "results_text": results_text,
        "results_combined": results_combined,
    }

    with open(os.path.join(model_folder, "evaluation/metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Évaluation terminée et résultats sauvegardés.")
    print(json.dumps(results, indent=4))

    return results