
from datasets import load_dataset

dataset = load_dataset("google/civil_comments")

print(dataset)
print(dataset["train"][0])

from datasets import load_dataset
import pandas as pd

dataset = load_dataset("google/civil_comments")

df = dataset["train"].to_pandas()

df = df[["text", "toxicity"]]

df["label"] = (df["toxicity"] >= 0.5).astype(int)

df = df[["text", "label"]]

df = df.dropna()

df.to_csv("comments.csv", index=False)

print("Dataset saved successfully!")
print("Total messages:", len(df))
print(df["label"].value_counts())
