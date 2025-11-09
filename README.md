# 🧠 Fine-Tuning d’un Modèle de Sentiment avec PEFT (LoRA)

Ce projet démontre comment fine-tuner un modèle pré-entraîné pour l’**analyse de sentiment** à l’aide de techniques modernes de **PEFT (Parameter-Efficient Fine-Tuning)** et plus précisément **LoRA (Low-Rank Adaptation)**.  
L'objectif est d’adapter `distilbert-base-uncased` sur le dataset **IMDB** pour classifier des critiques de films en *positif* ou *négatif*.

---

## 🚀 Contexte Technologique

### 🔍 Qu’est-ce que le Fine-Tuning ?
Le fine-tuning consiste à adapter un grand modèle (comme BERT) à une tâche spécifique. On continue l'entraînement du modèle sur un dataset ciblé afin de lui apprendre une compétence précise sans repartir de zéro.

### ⚙️ Pourquoi PEFT et LoRA ?
Fine-tuner tous les paramètres d’un modèle complet est coûteux.  
PEFT permet de **geler le modèle** et de **n’entraîner qu’une petite couche d’adaptation (LoRA)**.

| Approche | Avantages |
|----------|----------|
| **LoRA (PEFT)** | 🔹 0.1% des paramètres entraînés<br>🔹 Entraînement + rapide<br>🔹 Moins de mémoire GPU<br>🔹 Pas d’oubli des connaissances |

---

## 📦 Installation

### 1️⃣ Prérequis
- Python 3.8+
- `pip` ou `conda`

### 2️⃣ Installation du projet

```bash
# Cloner le dépôt
git clone [URL_DE_VOTRE_DEPOT_GITHUB]
cd [NOM_DU_DEPOT]
```
📌 Le fichier requirements.txt doit contenir :
transformers
datasets
evaluate
peft
torch
```bash
# Installer les dépendances
pip install -r requirements.txt
```
▶️ Exécution
python fine_tune_sentiment.py

🧪 Pipeline du Script fine_tune_sentiment.py
Étape	Description
1️⃣ Chargement des données	IMDB dataset via datasets
2️⃣ Modèle de base	distilbert-base-uncased + Tokenizer
3️⃣ Configuration LoRA	Création d’un LoraConfig
4️⃣ Prétraitement	Tokenisation des critiques
5️⃣ Entraînement (Trainer)	Fine-tuning PEFT (LoRA uniquement)
6️⃣ Évaluation	Accuracy sur le set de test
7️⃣ Inférence	Prédictions sur phrases nouvelles
📊 Résultats Attendus
Final evaluation results:
{'eval_loss': 0.35, 'eval_accuracy': 0.85, ...}


Exemples de prédictions :

Review: "This movie was fantastic!"
Prediction: POSITIVE (0.99)

Review: "I was disappointed..."
Prediction: NEGATIVE (0.99)



