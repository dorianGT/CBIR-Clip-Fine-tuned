# CBIR-Clip-Fine-tuned
Projet de détection de *near-duplicates* dans une base d’images historiques à l’aide de CLIP fine-tuné et d’une recherche de similarité basée sur FAISS.

## Objectif

Ce projet vise à identifier automatiquement les near duplicates dans une base d’images historiques. Il repose sur l’utilisation de **CLIP**, un modèle de vision et de langage, ajusté à l’aide d’un fine-tuning supervisé avec des couples *image / description* (caption).

## Méthodologie

1. **Fine-tuning de CLIP**  
   Le modèle CLIP est adapté au domaine historique en l'entraînant sur un ensemble d’images annotées avec leurs captions.

2. **Génération d’embeddings**  
   Une fois entraîné, toutes les images de la base sont encodées via CLIP.

3. **Indexation avec FAISS**  
   Les vecteurs sont indexés avec **FAISS** pour permettre une recherche rapide par similarité (utilisation de la distance **L2**).

4. **Recherche de voisins proches**  
   Pour chaque image, les **10 voisins les plus proches** sont extraits pour identifier les near duplicates.

## Installation

1. **Cloner le dépôt**

```
git clone https://github.com/ton-utilisateur/CBIR-Clip-Fine-tuned.git
cd CBIR-Clip-Fine-tuned
````

2. **Créer un environnement virtuel**

```
python -m venv venv
# Sous Windows
venv\Scripts\activate
# Sous MacOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```
pip install -r requirements.txt
```

## Comment exécuter le projet

### Prérequis

- Avoir un dossier contenant :
  - Les **images historiques** (`historicaldataset/`)
  - Le fichier Excel de **vérité terrain** (`lipade_images_similaires.xlsx`)
  - Le fchier csv contenant les **captions** (`captions.csv`)
- Chemins d'accès adaptés aux fichiers sur votre machine.
- Les dépendances Python installées (voir section Installation).

### Exécution étape par étape

1. **Nettoyage du dossier d’images**

```
python dataset_cleaning.py --image_folder "historicaldataset/"
````

2. **Génération du fichier regrouppant les informations (dvision train/val/test, associaton captions...) à partir de la vérité terrain**

```
python parse_groundtruth.py ^
  --image_folder "historicaldataset/" ^
  --excel_file "lipade_images_similaires.xlsx" ^
  --csv_descriptions_file "captions.csv"
```

3. **Entraînement du modèle CLIP + génération des embeddings + recherche FAISS + évaluation**

```
python main.py ^
  --model_folder "runs_02/exp01" ^
  --image_folder "historicaldataset/" ^
  --load_method clip ^
  --do_finetune True ^
  --epochs 20 ^
  --batch_size 64 ^
  --patience 5 ^
  --learning_rate 0.0001
```


## Évaluation et Visualisation

### Métriques utilisées

Pour évaluer les performances de la recherche de similarité, plusieurs métriques sont utilisées :

* **Recall\@K** : proportion des vrais *near duplicates* retrouvés dans les K voisins les plus proches.
* **Precision\@K** : proportion des images réellement similaires parmi les K voisins proposés.
* **F1\@K** : moyenne harmonique entre precision\@K et recall\@K.
* **mAP\@K (mean Average Precision)** : moyenne des précisions à chaque rang K où une image pertinente est retrouvée.
* **Seuil de distance (threshold)** : un seuil de distance L2 est appris (moyenne,médian etc des distances) pour classer les images comme similaires ou non.
* **Precision, Recall, F1 au seuil appris** : scores obtenus basée sur le seuil optimal.

### Visualisation des embeddings

Pour mieux comprendre la structure des embeddings générés par CLIP, deux techniques de **réduction de dimensionnalité** sont utilisées :

* **UMAP (Uniform Manifold Approximation and Projection)** : conserve la structure globale tout en projetant en 2D.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)** : met l'accent sur la préservation des proximités locales.

Ces visualisations permettent d’analyser :

* La séparation entre clusters d’images similaires.
* L’impact du fine-tuning sur la structure des embeddings.

### Meilleur Résultats

```
{
    "recall@1": 0.5275,
    "recall@5": 0.611,
    "recall@10": 0.633,
    "precision@1": 0.5275,
    "precision@5": 0.3089,
    "precision@10": 0.1921,
    "mAP@5": 0.3466,
    "mAP@10": 0.3793,
    "f1@5": 0.3036,
    "f1@10": 0.2416,
    "threshold": 0.5281,
    "precision_thresh": 0.4355,
    "recall_thresh": 0.3284,
    "f1_thresh": 0.3745
}
```

## Auteur

Projet développé dans le cadre du cours Modélisation de système de vision du master 2 VMI par Dorian GROUTEAU, supervisé par Camille KURTZ.
Autres contributions : Noureddine BERTRAND (autre méthode, même objectif) et Samuel GONCALVES.
