import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

# ================= UI HEADER =================
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("💬 Sentiment Analysis LSTM")
st.write("Masukkan teks untuk dianalisis")

# ================= LOAD MODEL =================
@st.cache_resource
def load_all():
    model = load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_all()

# ================= PREPROCESS =================
try:
    stop_words = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def preprocess(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=100)

# ================= INPUT =================
user_input = st.text_area("Input Text:")

# ================= BUTTON =================
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Masukkan teks dulu!")
    else:
        data = preprocess(user_input)
        pred = model.predict(data)[0][0]

        if pred > 0.3:
            st.success(f"Positif 😊 ({pred:.4f})")
        else:
            st.error(f"Negatif 😡 ({pred:.4f})")