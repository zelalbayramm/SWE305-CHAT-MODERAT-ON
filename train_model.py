
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

df = pd.read_csv("comments.csv")
df = df.dropna()

clean = df[df["label"] == 0].sample(n=5000, random_state=42)
offensive = df[df["label"] == 1].sample(n=5000, random_state=42)

data = pd.concat([clean, offensive]).sample(
    frac=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

X_test.to_frame().assign(label=y_test).to_csv(
    "test_data.csv", index=False
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print("Macro F1:", f1_score(y_test, predictions, average="macro"))

joblib.dump(model, "baseline_model.pkl")

print("Model saved successfully!")
