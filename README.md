🧠 Fine-Tuning d’un Modèle de Sentiment avec PEFT (LoRA)

Ce projet montre comment spécialiser un modèle de langage pré-entraîné pour une tâche de classification de sentiment, en utilisant une approche moderne et efficace : PEFT (Parameter-Efficient Fine-Tuning) avec LoRA (Low-Rank Adaptation).
L’objectif est de fine-tuner DistilBERT sur le dataset IMDB pour déterminer si une critique de film est positive ou négative.

🚀 Contexte Technologique
🎯 Qu’est-ce que le Fine-Tuning ?

Le fine-tuning consiste à adapter un grand modèle pré-entraîné (comme BERT) à une tâche spécifique en continuant son entraînement sur un petit dataset spécialisé. Cela permet de transférer ses connaissances générales vers une compétence ciblée (ex : analyse de sentiment).

⚙️ Pourquoi PEFT et LoRA ?

Fine-tuner tous les paramètres d’un modèle complet est coûteux en GPU. PEFT résout ce problème en gelant la majorité du modèle et en n’entraînant que de petites couches supplémentaires.

Méthode	Avantages
PEFT (LoRA)	➤ 0.1% de paramètres entraînés
➤ Très faible consommation GPU
➤ Modèle plus rapide & léger
📦 Installation
1️⃣ Prérequis

Python 3.8+

pip ou conda

2️⃣ Installation du projet
# Cloner le dépôt
git clone [URL_DE_VOTRE_DEPOT_GITHUB]
cd [NOM_DU_DEPOT]

# Installer les dépendances
pip install -r requirements.txt


🔍 Le fichier requirements.txt doit contenir :
transformers, datasets, evaluate, peft, torch

▶️ Exécution
python fine_tune_sentiment.py

🧪 Pipeline du Script fine_tune_sentiment.py
Étape	Description
1. Chargement des données	Dataset IMDB via datasets
2. Initialisation du modèle	distilbert-base-uncased + Tokenizer
3. Configuration LoRA	Définition d'une LoraConfig (r, alpha, dropout)
4. Tokenisation	Préparation des critiques en entrée modèle
5. Entraînement (Trainer)	Fine-tuning des adaptateurs LoRA uniquement
6. Évaluation	Calcul de la précision (accuracy)
7. Inférence	Prédictions sur de nouvelles phrases
📊 Résultats Attendus

Après entraînement, vous devriez voir :

Final evaluation results:
{'eval_loss': 0.35, 'eval_accuracy': 0.85, ...}


Prédictions exemple :

Review: "This movie was absolutely fantastic!"
Prediction: [{'label': 'POSITIVE', 'score': 0.99}]

Review: "I was really disappointed by this film."
Prediction: [{'label': 'NEGATIVE', 'score': 0.99}]

🧭 Pourquoi ce Projet est Important ?

✅ Comprendre les méthodes modernes de fine-tuning (PEFT)
✅ Réduire les coûts GPU tout en conservant les performances
✅ Préparer le terrain pour appliquer LoRA sur des LLMs (ChatGPT, LLaMa, Mistral)

🛡️ Licence

Ce projet est sous licence MIT — libre à vous de le modifier et l'adapter !

🎯 Prêt(e) à fine-tuner des LLM avec LoRA ? Ce projet est votre point de départ.
