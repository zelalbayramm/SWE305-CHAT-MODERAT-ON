
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

torch.manual_seed(42)

# Load training data
df = pd.read_csv("comments.csv").dropna()
test = pd.read_csv("test_data.csv").dropna()

# Recreate the original training split
clean = df[df["label"] == 0].sample(n=5000, random_state=42)
offensive = df[df["label"] == 1].sample(n=5000, random_state=42)

data = pd.concat([clean, offensive]).sample(
    frac=1, random_state=42
)

from sklearn.model_selection import train_test_split

X_train, _, y_train, _ = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

vectorizer = TfidfVectorizer(max_features=3000)

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(test["text"]).toarray()

y_train = torch.tensor(
    y_train.to_numpy(), dtype=torch.long
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Feedforward neural network
model = nn.Sequential(
    nn.Linear(3000, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

# Evaluation
model.eval()

with torch.no_grad():
    predictions = model(X_test).argmax(dim=1).numpy()

print(classification_report(test["label"], predictions))
print(
    "Macro F1:",
    f1_score(test["label"], predictions, average="macro")
)

torch.save(model.state_dict(), "neural_model.pth")
joblib.dump(vectorizer, "neural_vectorizer.pkl")

print("Neural Network saved successfully!")
