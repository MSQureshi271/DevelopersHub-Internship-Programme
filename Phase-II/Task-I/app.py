import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Tech"
}

@st.cache_resource
def load_model():
    model_path = "./bert-ag-news-final"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

def predict_topic(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()

    all_scores = {
        LABELS[i]: probabilities[0][i].item()
        for i in range(len(LABELS))
    }

    return LABELS[predicted_class_id], confidence, all_scores

st.title("📰 News Topic Classifier using BERT")

st.write(
    "This application classifies news articles into one of the following categories: World, Sports, Business, and Science/Tech. "
    "Enter a news article below and click 'Classify' to see the predicted topic."
)

st.markdown(
    """
    **Categories:**
    - World
    - Sports
    - Business
    - Sci/Tech
    """
)

user_input = st.text_area(
    "Enter the news article text here:",
    placeholder="Type or paste the news article text..."
)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        predicted_label, confidence, all_scores = predict_topic(user_input)

        st.subheader("Prediction Result")

        st.success(f"Predicted Topic: **{predicted_label}**")
        st.write(f"Confidence: **{confidence:.2%}**")

        st.subheader("Class Probabilities")

        for label, score in all_scores.items():
            st.write(f"{label}: {score:.2%}")
            st.progress(score)