import json
import os
import re

import joblib
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
from scipy.sparse import hstack

st.set_page_config(page_title="Smart MCQ Solver", page_icon="📝")

# ------------------------------------------------------------------
# Model artifacts hosted on Hugging Face Hub (downloaded on first run)
# ------------------------------------------------------------------
HF_BASE_URL = "https://huggingface.co/Shaurya-Sri/Smart-mcq-solver/resolve/main"
ARTIFACT_FILES = [
    "word_vectorizer.pkl",
    "char_vectorizer.pkl",
    "best_model.pt",
    "model_config.json",
]


def download_artifacts():
    for filename in ARTIFACT_FILES:
        if not os.path.exists(filename):
            url = f"{HF_BASE_URL}/{filename}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)

# ------------------------------------------------------------------
# Same model definition used in training — must match exactly
# ------------------------------------------------------------------
class DeepMCQNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------------
# Load artifacts once and cache them
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    download_artifacts()

    word_vectorizer = joblib.load("word_vectorizer.pkl")
    char_vectorizer = joblib.load("char_vectorizer.pkl")

    with open("model_config.json") as f:
        config = json.load(f)

    model = DeepMCQNet(config["input_dim"])
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()

    return word_vectorizer, char_vectorizer, model


word_vectorizer, char_vectorizer, model = load_artifacts()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📝 Smart MCQ Solver")
st.caption("TF-IDF (word + char) features → MLP (DeepMCQNet) scores each option.")

prompt = st.text_area("Question", placeholder="Type the MCQ question here...")

col1, col2 = st.columns(2)
with col1:
    option_a = st.text_input("Option A")
    option_b = st.text_input("Option B")
    option_c = st.text_input("Option C")
with col2:
    option_d = st.text_input("Option D")
    option_e = st.text_input("Option E")

options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d, "E": option_e}

if st.button("Solve", type="primary"):
    filled = {k: v for k, v in options.items() if v.strip()}

    if not prompt.strip() or len(filled) < 2:
        st.warning("Enter a question and at least two options.")
    else:
        prompt_clean = clean_text(prompt)
        texts, names = [], []
        for name, text in filled.items():
            pair = prompt_clean + " [SEP] " + clean_text(text)
            texts.append(pair)
            names.append(name)

        X_word = word_vectorizer.transform(texts)
        X_char = char_vectorizer.transform(texts)
        X = hstack([X_word, X_char])
        X_dense = torch.FloatTensor(X.toarray())

        with torch.no_grad():
            logits = model(X_dense)
            probs = torch.sigmoid(logits).numpy().flatten()

        ranked = sorted(zip(names, probs), key=lambda x: x[1], reverse=True)

        st.subheader("Ranked predictions")
        for i, (name, score) in enumerate(ranked, start=1):
            label = "🏆 Top pick" if i == 1 else f"#{i}"
            st.write(f"{label} — **Option {name}**: {options[name]}  ·  confidence {score:.3f}")

        top3 = " ".join(n for n, _ in ranked[:3])
        st.info(f"MAP@3 style submission string: `{top3}`")
