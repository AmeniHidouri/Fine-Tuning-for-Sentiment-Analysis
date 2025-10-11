# -*- coding: utf-8 -*-
"""
Script pour le fine-tuning d'un modèle d'analyse de sentiment (IMDb)
en utilisant une méthode de Parameter-Efficient Fine-Tuning (PEFT) avec LoRA.
"""

# Étape 1: Importer les bibliothèques nécessaires
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def main():
    """
    Fonction principale pour exécuter le workflow de fine-tuning.
    """
    # Étape 2: Charger le jeu de données
    print("Chargement du jeu de données IMDb...")
    dataset = load_dataset("imdb")
    
    # Pour une démonstration plus rapide, nous utilisons un sous-ensemble
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
    print("Jeu de données chargé et échantillonné.")

    # Étape 3: Charger le tokenizer et le modèle de base
    model_name = "distilbert-base-uncased"
    print(f"Chargement du tokenizer et du modèle pour '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("Tokenizer et modèle chargés.")

    # Étape 4: Configurer PEFT (LoRA)
    print("Configuration de LoRA...")
    lora_config = LoraConfig(
        r=16,                   # Rang des matrices d'adaptation
        lora_alpha=32,          # Facteur d'échelle pour les poids LoRA
        lora_dropout=0.1,       # Taux de dropout pour les couches LoRA
        bias="none",            # On n'entraîne pas les biais pour plus d'efficacité
        task_type="SEQ_CLS"     # Tâche de classification de séquence
    )

    # Envelopper le modèle de base avec les adaptateurs PEFT
    peft_model = get_peft_model(model, lora_config)
    print("LoRA configuré. Paramètres entraînables :")
    peft_model.print_trainable_parameters()

    # Étape 5: Prétraiter les données
    def tokenize_function(examples):
        """Fonction pour tokenizer le texte."""
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenisation des jeux de données...")
    tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = small_test_dataset.map(tokenize_function, batched=True)
    print("Tokenisation terminée.")

    # Étape 6: Mettre en place la métrique d'évaluation
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Étape 7: Définir les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="peft_sentiment_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-4, # Un taux d'apprentissage courant pour LoRA
        weight_decay=0.01,
    )

    # Étape 8: Créer le Trainer et lancer le fine-tuning
    print("Initialisation du Trainer...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Démarrage du fine-tuning...")
    trainer.train()
    print("Fine-tuning terminé.")

    # Étape 9: Évaluer le modèle final
    print("\nÉvaluation du modèle...")
    results = trainer.evaluate()
    print("Résultats de l'évaluation finale :")
    print(results)

    # Étape 10: Faire une prédiction sur de nouvelles phrases
    print("\n--- Inférence sur de nouveaux exemples ---")
    from transformers import pipeline

    # Le pipeline gère automatiquement le modèle PEFT
    sentiment_pipeline = pipeline("sentiment-analysis", model=peft_model, tokenizer=tokenizer)

    new_review = "This movie was absolutely fantastic, the acting was superb!"
    prediction = sentiment_pipeline(new_review)
    print(f"Critique : '{new_review}'")
    print(f"Prédiction : {prediction}")

    another_review = "I was really disappointed with this film. It was boring and slow."
    prediction = sentiment_pipeline(another_review)
    print(f"Critique : '{another_review}'")
    print(f"Prédiction : {prediction}")

if __name__ == "__main__":
    main()
