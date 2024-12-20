"""
@author: Moussa Kalla
"""

from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Initialisation de Flask
app = Flask(__name__)

# Charger le fichier CSV
data_path = "C:\\Python\\BERT Fine-Tuning pour la Classification des Intentions\\questions_intentions.csv"
df = pd.read_csv(data_path)

# Convertir les intentions en labels numériques
intent_map = {"Acheter": 0, "Vérifier disponibilité": 1, "Assistance": 2}
df['labels'] = df['intent'].map(intent_map)

# Diviser les données en ensembles d'entraînement et de validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['labels'], test_size=0.2, random_state=42
)

# Charger le tokenizer et le modèle pré-entraîné
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Préparer les données pour BERT
class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenization des textes
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

train_dataset = ChatbotDataset(train_encodings, list(train_labels))
val_dataset = ChatbotDataset(val_encodings, list(val_labels))

# Définir les paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no",
)

# Entraîner le modèle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Fonction pour prédire l'intention
def predict_intent(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return list(intent_map.keys())[prediction]

# Route principale pour afficher le formulaire et gérer les prédictions
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Récupérer la question soumise par l'utilisateur
        question = request.form["question"]
        # Faire une prédiction
        prediction = predict_intent(question)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
