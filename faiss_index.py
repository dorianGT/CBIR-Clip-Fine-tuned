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
    Recherche les k voisins les plus proches pour chaque requête dans l'index FAISS.

    Args:
        index (faiss.Index): Index FAISS dans lequel effectuer la recherche.
        queries (np.ndarray): Embeddings de requête.
        k (int): Nombre de résultats à retourner par requête.

    Returns:
        tuple: distances (np.ndarray), indices (np.ndarray)
    """
    faiss.normalize_L2(queries)
    distances, indices = index.search(queries, k)
    return distances, indices

def create_faiss_index(model_folder):
    """
    Crée un index FAISS à partir des embeddings sauvegardés dans un fichier .npz 
    et le sauvegarde dans le dossier du modèle.

    Args:
        model_folder (str): Chemin vers le dossier contenant le fichier 'embeddings.npz'.
                            Le fichier doit contenir les clés 'embeddings' et 'paths'.

    Effets de bord:
        - Sauvegarde un index FAISS sous le nom 'faiss.index' dans le même dossier.
        - Affiche un message indiquant le nombre de vecteurs indexés.
    """
    embeddings, paths = load_embeddings(os.path.join(model_folder, "embeddings.npz"))

    # Normaliser les embeddings avant de les ajouter à l'index FAISS
    embeddings = normalize_embeddings(embeddings)

    index = build_faiss_index(embeddings)

    save_index(index, os.path.join(model_folder, "faiss.index"))

    print(f"Index créé avec {len(paths)} vecteurs et sauvegardé dans {os.path.join(model_folder, 'faiss.index')}")

# def main():
#     """
#     Fonction principale pour créer un index FAISS à partir des embeddings
#     contenus dans le dossier 'model_folder'.
#     """
#     create_faiss_index("model_folder")

# if __name__ == "__main__":
#     main()