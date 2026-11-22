
import streamlit as st
import joblib

st.set_page_config(page_title="Chat Moderation")

@st.cache_resource
def load_model():
    return joblib.load("baseline_model.pkl")

model = load_model()

st.title("Real-Time Chat Moderation")
st.write("AI-Powered Message Classification")

message = st.text_area(
    "Enter your message:",
    key="user_message"
)

if st.button("Analyze Message"):
    if message.strip():
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
        confidence = probabilities[prediction] * 100

        if prediction == 0:
            st.success("Clean Message")
        else:
            st.error("Offensive Message")

        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a message.")
