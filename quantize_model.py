
import os
import torch
torch.backends.quantized.engine = "qnnpack"
from transformers import AutoModelForSequenceClassification

model_path = "transformer_model"

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

original_size = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
) / (1024 * 1024)

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_size = sum(
    p.numel() * p.element_size()
    for p in quantized_model.parameters()
) / (1024 * 1024)

print("Original model size:", round(original_size, 2), "MB")
print("Quantized parameter size:", round(quantized_size, 2), "MB")

torch.save(
    quantized_model.state_dict(),
    "quantized_model.pth"
)

print("Quantized model saved successfully!")
import torch
