
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.backends.quantized.engine = "qnnpack"
torch.set_num_threads(1)

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
    torch.load(
        "quantized_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
quantized_model.eval()

text = "Hello, how are you?"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

def measure_speed(model, inputs, runs=100):
    with torch.inference_mode():
        for _ in range(10):
            model(**inputs)

        start = time.perf_counter()

        for _ in range(runs):
            model(**inputs)

        end = time.perf_counter()

    return (end - start) / runs * 1000

original_time = measure_speed(model, inputs)
quantized_time = measure_speed(quantized_model, inputs)

print("Original Model:", round(original_time, 2), "ms")
print("Quantized Model:", round(quantized_time, 2), "ms")
print("Speedup:", round(original_time / quantized_time, 2), "x")
