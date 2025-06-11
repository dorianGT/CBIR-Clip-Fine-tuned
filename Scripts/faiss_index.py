import numpy as np
import faiss
import os

def load_embeddings(embedding_path):
    """
    Charge les embeddings et les chemins d'accès depuis un fichier .npz.

    Args:
        embedding_path (str): Chemin vers le fichier contenant les embeddings et les chemins.

    Returns:
        tuple: embeddings (np.ndarray), paths (list)
    """
    data = np.load(embedding_path, allow_pickle=True)
    embeddings = data["embeddings"].astype('float32')
    paths = data["paths"].tolist()
    return embeddings, paths

def normalize_embeddings(embeddings):
    """
    Normalise les embeddings pour utiliser la similarité cosinus avec FAISS (via la distance L2).

    Args:
        embeddings (np.ndarray): Tableau d'embeddings à normaliser.

    Returns:
        np.ndarray: Embeddings normalisés.
    """
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings):
    """
    Construit un index FAISS basé sur la distance L2 à partir des embeddings.

    Args:
        embeddings (np.ndarray): Embeddings normalisés.

    Returns:
        faiss.IndexFlatL2: Index FAISS construit.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Distance L2
    index.add(embeddings)  # Ajout des embeddings
    return index

def save_index(index, index_path):
    """
    Sauvegarde un index FAISS.

    Args:
        index (faiss.Index): Index FAISS à sauvegarder.
        index_path (str): Chemin de sauvegarde de l'index.
    """
    faiss.write_index(index, index_path)

def load_index_faiss(index_path):
    """
    Charge un index FAISS.

    Args:
        index_path (str): Chemin vers l'index sauvegardé.

    Returns:
        faiss.Index: Index FAISS chargé.
    """
    return faiss.read_index(index_path)

def search(index, queries, k=5):
    """
    Recherche les k voisins les plus proches pour chaque vecteur de requête dans l'index FAISS.

    Args:
        index (faiss.Index): Index FAISS dans lequel effectuer la recherche.
        queries (np.ndarray): Embeddings des requêtes de forme (M, D).
        k (int, optional): Nombre de voisins à retourner par requête (default: 5).

    Returns:
        tuple:
            - distances (np.ndarray): Distances L2 aux voisins (forme (M, k)).
            - indices (np.ndarray): Indices des voisins trouvés (forme (M, k)).
    """
    faiss.normalize_L2(queries)
    distances, indices = index.search(queries, k)
    return distances, indices

def create_faiss_index(model_folder):
    """
    Crée trois index FAISS (image, texte et combiné) à partir d'embeddings stockés dans le dossier spécifié.

    - embeddings_image.npz - faiss_image.index
    - embeddings_text.npz - faiss_text.index
    - embeddings_combined.npz - faiss_combined.index

    Chaque .npz doit contenir :
        - 'embeddings' : np.ndarray de forme (N, D)
        - 'paths' : liste de N chemins ou identifiants correspondants

    Args:
        model_folder (str): Chemin vers le dossier contenant les fichiers .npz.

    Effets:
        - Trois fichiers .index sont créés dans le dossier.
    """

    embeddings_image, paths = load_embeddings(os.path.join(model_folder, "embeddings_image.npz"))
    embeddings_text, _ = load_embeddings(os.path.join(model_folder, "embeddings_text.npz"))
    embeddings_combined, _ = load_embeddings(os.path.join(model_folder, "embeddings_combined.npz"))

    # Normaliser les embeddings avant de les ajouter à l'index FAISS
    embeddings_image = normalize_embeddings(embeddings_image)
    embeddings_text = normalize_embeddings(embeddings_text)
    embeddings_combined = normalize_embeddings(embeddings_text)

    index_image = build_faiss_index(embeddings_image)
    index_text = build_faiss_index(embeddings_text)
    index_combined = build_faiss_index(embeddings_combined)

    save_index(index_image, os.path.join(model_folder, "faiss_image.index"))
    save_index(index_text, os.path.join(model_folder, "faiss_text.index"))
    save_index(index_combined, os.path.join(model_folder, "faiss_combined.index"))

    print(f"Index créé avec {len(paths)} vecteurs et sauvegardé dans faiss_image.index, faiss_text.index et faiss_combined.index.")

# def main():
#     """
#     Fonction principale pour créer un index FAISS à partir des embeddings
#     contenus dans le dossier 'model_folder'.
#     """
#     create_faiss_index("model_folder")

# if __name__ == "__main__":
#     main()