
import streamlit as st
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
st.set_page_config(
    page_title="Real-Time Chat Moderation",
    page_icon="💬"
)

st.title("Real-Time Chat Moderation")
st.write("AI-Powered Message Classification")

@st.cache_resource
def load_baseline():
    return joblib.load("baseline_model.pkl")

@st.cache_resource
def load_neural():
    vectorizer = joblib.load("neural_vectorizer.pkl")

    model = nn.Sequential(
        nn.Linear(3000, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2)
    )

    model.load_state_dict(
        torch.load("neural_model.pth", map_location="cpu",
                   weights_only=True)
    )
    model.eval()
    return model, vectorizer

@st.cache_resource
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained(
        "transformer_model"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "transformer_model"
    )
    model.eval()
    return model, tokenizer
@st.cache_resource
def load_quantized():
    torch.backends.quantized.engine = "qnnpack"

    tokenizer = AutoTokenizer.from_pretrained("transformer_model")

    model = AutoModelForSequenceClassification.from_pretrained(
        "transformer_model"
    )

    model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    model.load_state_dict(
        torch.load(
            "quantized_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    return model, tokenizer

model_choice = st.selectbox(
    "Select AI Model",
    ["Baseline", "Neural Network", "Transformer", "Quantized Transformer"]
)

message = st.text_area(
    "Enter your message:",
    key="moderation_message"
)

if st.button("Analyze Message"):
    if message.strip():
        with st.spinner("Analyzing..."):
            start_time = time.perf_counter()
            if model_choice == "Baseline":
                model = load_baseline()
                prediction = int(model.predict([message])[0])
                probabilities = model.predict_proba([message])[0]
                confidence = float(max(probabilities)) * 100

            elif model_choice == "Neural Network":
                model, vectorizer = load_neural()
                features = vectorizer.transform([message]).toarray()
                tensor = torch.tensor(features, dtype=torch.float32)

                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    prediction = int(torch.argmax(probabilities))
                    confidence = float(probabilities[prediction]) * 100
            elif model_choice == "Transformer":
                model, tokenizer = load_transformer()
            else:
                model, tokenizer = load_quantized()
    
             
                inputs = tokenizer(
                    message,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(
                        outputs.logits, dim=1
                    )[0]
                    prediction = int(torch.argmax(probabilities))
                    confidence = float(probabilities[prediction]) * 100

        if prediction == 0:
            st.success("Clean Message")
        else:
            st.error("Offensive Message")

        st.write(f"Confidence: {confidence:.2f}%")
        st.info(f"Model Used: {model_choice}")
        elapsed_time = (time.perf_counter() - start_time) * 1000
        st.write(f"Response Time: {elapsed_time:.2f} ms")
        

    else:
        st.warning("Please enter a message.")
