
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

torch.backends.quantized.engine = "qnnpack"

model_path = "transformer_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_model.load_state_dict(
    torch.load("quantized_model.pth", map_location="cpu", weights_only=True)
)

quantized_model.eval()

test = pd.read_csv("test_data.csv")

predictions = []

for text in test["text"]:
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = quantized_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    predictions.append(prediction)

accuracy = accuracy_score(test["label"], predictions)
macro_f1 = f1_score(test["label"], predictions, average="macro")

print("Quantized Model Accuracy:", accuracy)
print("Quantized Model Macro F1:", macro_f1)
