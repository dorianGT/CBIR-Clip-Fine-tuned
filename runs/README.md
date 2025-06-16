# Expériences Fine-tuning

Ce dossier contient les différentes expériences de fine-tuning réalisées.

## Contenu des dossiers

- **runs_02** et **runs_03** : Contiennent des **tests** et expérimentations préliminaires.
- **runs_final** : Contient les **expérimentations principales résumées**.

## Contenu de `runs_final`

| Dossier    | Description                                                                                   |
|------------|----------------------------------------------------------------------------------------------|
| **baseline** | **Modèle CLIP sans fine-tuning**, utilisé comme référence de performance.                      |
| **exp01**    | **Fine-tuning gelé pour Clip** mais avec projecteurs (MLP) **image + texte** dégelés.  |
| **exp02**    | **Fine-tuning complet** sans projecteurs. Tout le modèle est dégélé.                           |
| **exp03**    | **Fine-tuning partiel** : seule **la dernière couche** est dégélée. Pas de projecteurs.        |
| **exp04**    | **Fine-tuning avec LoRA**. Tout le modèle est dégélé, mais uniquement des adaptations LoRA sont apprises. Pas de projecteurs. |
| **Fine Tuning Partiel**    | **Fine-tuning partiel**. Expérimentations plus poussé (hyperparametre, VeCLIP...). |

## Paramètres

- Dataset : `HighVision_Corpus_Groundtruth`
- Méthode de chargement : `clip` (ou `clip+lora` pour exp04)
- epochs : 20
- batch_size : 64 - 256
- patience : 5 (early stopping)
- learning_rate : 0.00001 - 0.0001
- Optimiseur : AdamW
- Scheduler : ReduceLROnPlateau (patience=2)
- Température (loss) : 0.07
