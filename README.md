# CBIR par CLIP Fine-tuné pour la Détection de Near-Duplicates dans les Archives Historiques

Projet de détection de *near-duplicates* dans une base d’images historiques à l’aide de CLIP fine-tuné.

## Objectif

Ce projet vise à détecter automatiquement les *near duplicates* (doublons visuellement très similaires) dans une base d’images historiques. Il repose sur l’utilisation de **CLIP**, un modèle multimodal vision/texte, adapté via un fine-tuning supervisé à partir de couples *image / description (caption)* spécifiques au domaine.

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

## Méthodologie

![Pipeline global](images/pipeline_global.png)

Le pipeline global se déroule en plusieurs étapes clés :

### 1. **Préparation du dataset**

* Nettoyage de la base d’images.
* Séparation des images en deux catégories : *singletons* (sans doublon) et groupes de *near-duplicates* (ND).
* Suppression des images sans métadonnées ou mal étiquetées.

### 2. **Amélioration des captions**

* Génération ou enrichissement des descriptions textuelles associées aux images via **VeCLIP**.

### 3. **Fine-tuning de CLIP**

* Ajustement supervisé du modèle CLIP sur les couples *image / caption* du dataset.
* Objectif : apprendre des représentations plus sensibles aux similarités contextuelles et historiques.

### 4. **Génération des embeddings**

* Calcul de trois types d’embeddings :

  * **Image embeddings** via l’encodeur visuel.
  * **Text embeddings** via l’encodeur textuel.
  * **Embeddings combinés** : moyenne pondérée des deux précédents.

### 5. **Indexation avec FAISS (optionnelle)**

* Création de trois index de recherche (image, texte, combiné) à l’aide de **FAISS** pour permettre une recherche rapide.

### 6. **Recherche de similarité**

* Pour une image requête, récupération des *top-k* candidats similaires :

  * En utilisant l’un des trois types d’embeddings.
  * Avec ou sans FAISS (avec pour une recherche rapide et scalable).
  * Utilisation de la distance L2 (normalisé) pour mesurer la similarité.

### 7. **Évaluation et comparaison**

* Évaluation des performances via des métriques adaptées : précision à *k*, rappel, MAP, etc.
* Comparaison entre les différents types d'embeddings et méthodes d’indexation.
  
## Le Dataset

Le dataset utilisé est constitué d’**images d’archives historiques** issues de diverses sources patrimoniales. Il est enrichi de plusieurs types d’informations :

* **Images** : photographies anciennes, numérisées à partir de fonds d’archives.
* **Captions** : descriptions générées automatiquement à l’aide du modèle **BLIP** (Bootstrapped Language Image Pretraining) par Samuel GONCALVES, puis raffinées avec VeCLIP dans les étapes suivantes du pipeline.
* **Métadonnées associées** :

  * Date (année, parfois approximative)
  * Lieu (ville, région, pays…)
  * Source / collection d’origine
  * Tags additionnels dans certains cas (thème, personne, événement…)

Voic quelques exemples de statistiques :

![Dataset](images/dataset.png)

## Amélioration des captions avec **VeCLIP**

### Problème

Les captions générées automatiquement (par exemple avec **BLIP**) sont souvent :

* Trop génériques (*"a black and white photo of a man"*)
* Incomplètes ou non spécifiques au contexte historique
* Peu utiles pour différencier des images visuellement similaires (*near-duplicates*)

### Solution : VeCLIP

Le pipeline **VeCLIP** vise à enrichir les descriptions textuelles en combinant plusieurs sources d’information :

* Les **captions générées**
* Les **métadonnées disponibles** (lieu, date, source…)

Cela permet de générer des **légendes plus riches, informatives et contextuelles**, mieux adaptées aux besoins de recherche d’images proches.

### Pipeline utilisé

<img src="images/pipeline_veclip.png" alt="Pipeline veclip" height="400"/>

## Fine-tuning de **CLIP**

L’adaptation du modèle **CLIP** au domaine des archives historiques est une étape cruciale du projet. Plusieurs stratégies de fine-tuning ont été envisagées pour ajuster ses représentations aux particularités du dataset.

### Stratégies de fine-tuning

#### **Fine-tuning complet**

Tous les paramètres du modèle CLIP (vision et texte) sont mis à jour durant l’entraînement.

#### **Fine-tuning partiel**

Seules les **dernières couches** des encodeurs visuel et textuel sont ajustées.

#### **LoRA (Low-Rank Adaptation)**

Méthode légère : insertion de **modules entraînables** dans les couches de CLIP, cela permet d’adapter le modèle avec très peu de paramètres modifiés.
  
#### **Projecteurs**

Ajout de deux MLP après les embeddings CLIP pour mieux adapter l’espace latent.

## Génération des embeddings & Indexation FAISS

### Génération des embeddings

Chaque image et sa description textuelle sont encodées séparément par le modèle **CLIP fine-tuné**.  
Leurs **embeddings sont moyennés** pour former un vecteur de représentation multimodal unique.

### Indexation avec FAISS

Les vecteurs sont ensuite **indexés avec FAISS** (optionnel) pour permettre une recherche rapide et efficace à grande échelle.  
La similarité est mesurée avec la **distance L2 normalisée**.  
Pour chaque image, on récupère ses **10 voisins les plus proches** (*top-k search*).

## Évaluation et Visualisation

### Métriques utilisées

Pour évaluer les performances de la recherche de similarité, plusieurs métriques sont utilisées :

* **Recall\@K** : proportion des vrais *near duplicates* retrouvés dans les K voisins les plus proches.
* **Precision\@K** : proportion des images réellement similaires parmi les K voisins proposés.
* **F1\@K** : moyenne harmonique entre precision\@K et recall\@K.
* **mAP\@K (mean Average Precision)** : moyenne des précisions à chaque rang K où une image pertinente est retrouvée.
* **Seuil de distance (threshold)** : un seuil de distance L2 est appris (moyenne,médian etc des distances des K voisins les plus proches) pour classer les images comme similaires ou non.
* **Precision, Recall, F1 au seuil appris** : scores obtenus basée sur le seuil optimal.

### Visualisation des embeddings

Pour mieux comprendre la structure des embeddings générés par CLIP, deux techniques de **réduction de dimensionnalité** sont utilisées :

* **UMAP (Uniform Manifold Approximation and Projection)** : conserve la structure globale tout en projetant en 2D.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)** : met l'accent sur la préservation des proximités locales.

Ces visualisations permettent d’analyser :

* La séparation entre clusters d’images similaires.
* L’impact du fine-tuning sur la structure des embeddings.

### Résultats

![Resultats1](images/resultats1.PNG)

![Resultats2](images/resultats2.PNG)

## Conclusion

## Auteur

Projet développé dans le cadre du cours Modélisation de système de vision du master 2 VMI par Dorian GROUTEAU, supervisé par Camille KURTZ.

Autres contributions : Noureddine BERTRAND (autre méthode, même objectif) et Samuel GONCALVES.
