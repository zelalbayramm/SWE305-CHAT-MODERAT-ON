
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

torch.manual_seed(42)

# Load data
df = pd.read_csv("comments.csv")
df = df[["text", "label"]].dropna()

# Balanced sample
clean = df[df["label"] == 0].sample(n=500, random_state=42)
offensive = df[df["label"] == 1].sample(n=500, random_state=42)

data = pd.concat([clean, offensive]).sample(
    frac=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    data["text"].tolist(),
    data["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Load pretrained Transformer
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenize text
train_tokens = tokenizer(
    X_train,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

test_tokens = tokenizer(
    X_test,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

train_labels = torch.tensor(y_train, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()

batch_size = 16

for epoch in range(2):
    total_loss = 0

    for i in range(0, len(train_labels), batch_size):
        batch = {
            key: value[i:i + batch_size]
            for key, value in train_tokens.items()
        }

        labels = train_labels[i:i + batch_size]

        optimizer.zero_grad()

        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/2 - Loss: {total_loss:.4f}")

# Evaluation
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(test_labels), batch_size):
        batch = {
            key: value[i:i + batch_size]
            for key, value in test_tokens.items()
        }

        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.tolist())

print(classification_report(y_test, predictions))
print("Macro F1:", f1_score(y_test, predictions, average="macro"))

# Save model
model.save_pretrained("transformer_model")
tokenizer.save_pretrained("transformer_model")

print("Transformer saved successfully!")
