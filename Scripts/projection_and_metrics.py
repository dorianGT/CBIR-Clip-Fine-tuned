import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.cm as cm
from distinctipy import distinctipy
import numpy as np

### Histrograms & Projections ###

def save_histogram_distance(D, save_path, title="Distance Histogram"):
    """
    Enregistre un histogramme des distances (hors self-match) à partir des résultats de recherche FAISS.

    Args:
        D (np.ndarray): Matrice des distances obtenue par FAISS, où D[i, j] est la distance entre l’élément i et son j-ième plus proche voisin.
        save_path (str): Chemin de sauvegarde de l'image générée.
        title (str): Titre du graphique (par défaut "Distance Histogram").

    Returns:
        None
    """
    plt.figure(figsize=(8,6))
    plt.hist(D[:, 1:].flatten(), bins=50)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_projection(embeddings, save_path, method="tsne"):
    """
    Sauvegarde une projection 2D (t-SNE ou UMAP) des embeddings sans coloration de groupe.

    Args:
        embeddings (np.ndarray): Matrice des vecteurs d’embedding à projeter.
        save_path (str): Chemin de sauvegarde de l'image générée.
        method (str): Méthode de réduction de dimension, "tsne" ou "umap" (par défaut "tsne").

    Returns:
        None
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "Projection t-SNE des embeddings"
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        title = "Projection UMAP des embeddings"

    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12,8))
    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_colored_projection(embeddings, paths, groups, save_path, method="tsne"):
    """
    Sauvegarde une projection 2D (t-SNE ou UMAP) des embeddings avec une coloration selon les groupes fournis.

    Args:
        embeddings (np.ndarray): Matrice des vecteurs d’embedding à projeter.
        paths (List[str]): Liste des chemins d’accès aux fichiers correspondant aux embeddings.
        groups (List[List[str]]): Liste de groupes de chemins représentant les objets similaires.
        save_path (str): Chemin de sauvegarde de l'image générée.
        method (str): Méthode de réduction de dimension, "tsne" ou "umap" (par défaut "tsne").

    Returns:
        None
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "Projection t-SNE des embeddings"
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        title = "Projection UMAP des embeddings"

    embeddings_2d = reducer.fit_transform(embeddings)

    path_to_group_id = {}
    for i, group in enumerate(groups):
        for path in group:
            path_to_group_id[path] = i

    num_groups = len(groups)
    group_colors = distinctipy.get_colors(num_groups)
    default_color = (0.6, 0.6, 0.6)  # gris pour les singletons

    colors = []
    for path in paths:
        group_id = path_to_group_id.get(path, None)
        if group_id is not None:
            colors.append(group_colors[group_id])
        else:
            colors.append(default_color)

    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=6, alpha=0.7)
    plt.title(title)
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

### METRICS NO FAISS ###

def compute_top_k_neighbors(embeddings: np.ndarray, k: int, query_idx: int):
    query_embedding = embeddings[query_idx]
    # Calcul des distances euclidiennes à tous les autres vecteurs
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    # On exclut l'élément lui-même
    distances[query_idx] = np.inf
    # On récupère les indices des k plus proches voisins
    return np.argsort(distances)[:k]


def recall_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule le Recall@K sur des groupes d'images similaires (near duplicates).

    Le Recall@K mesure la proportion d'éléments pertinents (du même groupe que l'image requête)
    retrouvés parmi les K plus proches voisins.

    Formule :
        Recall@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / (Nombre total d'éléments pertinents)

    Args:
        embeddings (np.ndarray): Tableau des vecteurs d'embedding (taille N x D).
        paths (List[str]): Liste des chemins d'images correspondant aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins à considérer (top-k).

    Returns:
        float: Moyenne du Recall@K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_recall = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_possible = len(relevant_items)

            if total_possible > 0:
                total_recall += retrieved_relevant / total_possible
                count += 1

    return total_recall / count if count > 0 else 0.0


def precision_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule la Precision@K sur les groupes d'images similaires.

    La Precision@K mesure la proportion des K éléments retournés qui sont réellement
    pertinents (du même groupe que l'image requête).

    Formule :
        Precision@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / K

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne de la précision à K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_precision = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_precision += retrieved_relevant / k
            count += 1

    return total_precision / count if count > 0 else 0.0


def mean_average_precision_no_faiss(embeddings, paths, groups, k=10):
    """
    Calcule le Mean Average Precision (mAP) pour des groupes d'images similaires.

    Le mAP tient compte de la position des bons résultats parmi les K premiers.

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires.
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne des précisions moyennes (mAP) sur toutes les requêtes.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}
    average_precisions = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            relevant_hits = 0
            precision_sum = 0.0

            for rank, retrieved_path in enumerate(retrieved_paths, start=1):
                if retrieved_path in relevant_items:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank

            if relevant_hits > 0:
                average_precisions.append(precision_sum / len(relevant_items))
            else:
                average_precisions.append(0.0)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def f1_score_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule le F1-Score@K, moyenne harmonique entre la Precision@K et le Recall@K.

    Formule :
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne du F1@K sur toutes les requêtes valides.
    """
    precision = precision_at_k_no_faiss(embeddings, paths, groups, k)
    recall = recall_at_k_no_faiss(embeddings, paths, groups, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def evaluate_with_distance_no_faiss(embeddings, paths, groups, k=10, threshold_type="mean"):
    """
    Évalue la performance du système de recherche d'images par similarité en utilisant un seuil sur la distance,
    sans FAISS (utilise uniquement numpy pour les calculs).

    Args:
        embeddings (np.ndarray): Embeddings des images (shape: [N, D]).
        paths (List[str]): Chemins des images correspondant aux embeddings.
        groups (List[List[Dict]]): Groupes d'images similaires (ground truth) sous forme [{image, caption}, ...].
        k (int): Nombre de voisins à rechercher pour chaque image (k+1 en réalité car on ignore l'image elle-même).
        threshold_type (str): Méthode pour calculer le seuil de distance. Options : "mean", "median", "percentile_25".

    Returns:
        threshold (float): Seuil utilisé pour filtrer les voisins.
        precision (float): Précision globale (TP / (TP + FP)).
        recall (float): Rappel global (TP / (TP + FN)).
        f1 (float): Score F1 global.
    """
    
    groups_image_names = [[img['image'] if isinstance(img, dict) else img for img in group] for group in groups]
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups_image_names for path in group}

    # Collecter les distances pour calculer le seuil global
    all_distances = []

    for i in range(len(embeddings)):
        query_embedding = embeddings[i]
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        distances[i] = np.inf  # ignore self-distance
        all_distances.extend(distances[np.isfinite(distances)])

    all_distances = np.array(all_distances)

    # Calcul du seuil
    if threshold_type == "mean":
        threshold = np.mean(all_distances)
    elif threshold_type == "median":
        threshold = np.median(all_distances)
    elif threshold_type == "percentile_25":
        threshold = np.percentile(all_distances, 25)
    else:
        raise ValueError(f"Unsupported threshold_type: {threshold_type}")
    
    TP = 0
    FP = 0
    FN = 0

    for group in groups_image_names:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_embedding = embeddings[query_idx]
            query_group = set(group) - {query_path}

            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            distances[query_idx] = np.inf

            neighbor_indices = np.argsort(distances)[:k]
            retrieved_paths = [
                paths[i]
                for i in neighbor_indices
                if distances[i] < threshold and paths[i] != query_path
            ]
            retrieved_set = set(retrieved_paths)

            tp = len(retrieved_set & query_group)
            fp = len(retrieved_set - query_group)
            fn = len(query_group - retrieved_set)

            TP += tp
            FP += fp
            FN += fn

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(threshold), float(precision), float(recall), float(f1)


### METRICS FAISS ###

def recall_at_k(index, embeddings, paths, groups, k=5):
    """
    Calcule le Recall@K sur des groupes d'images similaires (near duplicates).

    Le Recall@K mesure la proportion d'éléments pertinents (du même groupe que l'image requête)
    retrouvés parmi les K plus proches voisins.

    Formule :
        Recall@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / (Nombre total d'éléments pertinents)

    Args:
        index: Index FAISS utilisé pour la recherche des plus proches voisins.
        embeddings (np.ndarray): Tableau des vecteurs d'embedding (taille N x D).
        paths (List[str]): Liste des chemins d'images correspondant aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins à considérer (top-k).

    Returns:
        float: Moyenne du Recall@K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_recall = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_possible = len(relevant_items)

            if total_possible > 0:
                total_recall += retrieved_relevant / total_possible
                count += 1

    return total_recall / count if count > 0 else 0.0

def precision_at_k(index, embeddings, paths, groups, k=5):
    """
    Calcule la Precision@K sur les groupes d'images similaires.

    La Precision@K mesure la proportion des K éléments retournés qui sont réellement
    pertinents (du même groupe que l'image requête).

    Formule :
        Precision@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / K

    Args:
        index: Index FAISS utilisé pour la recherche.
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne de la précision à K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_precision = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path][:k]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_precision += retrieved_relevant / k
            count += 1

    return total_precision / count if count > 0 else 0.0



def mean_average_precision(index, embeddings, paths, groups, k=10):
    """
    Calcule le Mean Average Precision (mAP) pour des groupes d'images similaires.

    Le mAP tient compte de la position des bons résultats parmi les K premiers.

    Args:
        index: Index Faiss utilisé pour la recherche.
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires.
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne des précisions moyennes (mAP) sur toutes les requêtes.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}
    average_precisions = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]

            relevant_hits = 0
            precision_sum = 0.0

            # Calcul de la précision moyenne
            for rank, retrieved_path in enumerate(retrieved_paths, start=1):
                if retrieved_path in query_group:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank

            if relevant_hits > 0:
                average_precisions.append(precision_sum / len(query_group))
            else:
                average_precisions.append(0.0)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def f1_score_at_k(index, embeddings, paths, groups, k=5):
    """
    Calcule le F1-Score@K, moyenne harmonique entre la Precision@K et le Recall@K.

    Formule :
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        index: Index FAISS utilisé pour la recherche.
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne du F1@K sur toutes les requêtes valides.
    """
    precision = precision_at_k(index, embeddings, paths, groups, k)
    recall = recall_at_k(index, embeddings, paths, groups, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def evaluate_with_distance_threshold(index, embeddings, paths, groups, k=10, threshold_type="mean"):
    """
    Évalue la performance du système de recherche d'images par similarité en utilisant un seuil sur la distance
    pour filtrer les voisins.

    Paramètres :
        index (faiss.Index): Index FAISS pour la recherche.
        embeddings (np.ndarray): Embeddings des images.
        paths (List[str]): Chemins des images correspondant aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins à rechercher pour chaque image (k+1 en réalité car on ignore l'image elle-même).
        threshold_type (str): Méthode pour calculer le seuil de distance. 
            Options : "mean", "median", "percentile_25".

    Retourne :
        threshold (float): Seuil utilisé pour filtrer les voisins.
        precision (float): Précision globale (TP / (TP + FP)).
        recall (float): Rappel global (TP / (TP + FN)).
        f1 (float): Score F1 global.
    """
    
    # Création d'un dictionnaire pour accéder rapidement à l'index et au groupe de chaque image
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    # Recherche des k+1 plus proches voisins pour chaque image
    D, _ = index.search(embeddings.astype(np.float32), k + 1)

    # Calcul du seuil de distance selon la stratégie choisie
    if threshold_type == "mean":
        threshold = np.mean(D[:, 1:])  # on ignore la première colonne (distance à soi-même)
    elif threshold_type == "median":
        threshold = np.median(D[:, 1:])
    elif threshold_type == "percentile_25":
        threshold = np.percentile(D[:, 1:], 25)
    else:
        raise ValueError("Unsupported threshold_type")

    # Initialisation des compteurs pour TP, FP, FN
    TP = 0
    FP = 0
    FN = 0

    # Boucle sur chaque image de chaque groupe
    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue  # sécurité si le chemin est manquant

            query_idx = path_to_index[query_path]
            # On considère toutes les autres images du même groupe comme les vraies positives attendues
            query_group = set(path_to_group[query_path]) - {query_path}

            # Recherche des k+1 voisins les plus proches de l'image requête
            D_query, I_query = index.search(
                np.array([embeddings[query_idx]]).astype(np.float32), k + 1
            )

            # On filtre les voisins : on garde ceux sous le seuil de distance et différents de la requête
            neighbors = [
                (paths[i], d)
                for i, d in zip(I_query[0], D_query[0])
                if paths[i] != query_path and d < threshold
            ]

            # On récupère uniquement les chemins des voisins sélectionnés
            retrieved_paths = [p for p, _ in neighbors]
            retrieved_set = set(retrieved_paths)

            # Calcul des vrais positifs (retrouvés et pertinents), faux positifs et faux négatifs
            tp = len(retrieved_set & query_group)
            fp = len(retrieved_set - query_group)
            fn = len(query_group - retrieved_set)

            TP += tp
            FP += fp
            FN += fn

    # Calcul des métriques globales
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    threshold = float(threshold)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    return threshold, precision, recall, f1
