import os
import re
import pandas as pd
import joblib
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Paths
DATA_PATH = "dataset/resume_data.csv"
MODEL_PATH = "models/resume_classifier_bert.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
BERT_EMBEDDINGS_PATH = "models/bert_embeddings.pkl"

# Create model directory
os.makedirs("models", exist_ok=True)

# Load BERT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_MODEL = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

# Text Cleaning Function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text)).strip().lower()
    return text

# Convert Text to BERT Embeddings
def get_bert_embedding(text):
    tokens = TOKENIZER(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = BERT_MODEL(**tokens)
    return output.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# Train Model
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("🚀 Training new BERT model...")

    df = pd.read_csv(DATA_PATH)
    df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

    # Encode Labels
    label_encoder = LabelEncoder()
    df["Category_Encoded"] = label_encoder.fit_transform(df["Category"])

    # Generate BERT Embeddings
    resume_embeddings = []
    print("🔄 Generating BERT embeddings...")
    for text in tqdm(df["Cleaned_Resume"], desc="Processing Resumes"):
        resume_embeddings.append(get_bert_embedding(text))
    
    # Save embeddings
    joblib.dump(resume_embeddings, BERT_EMBEDDINGS_PATH)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    print("🔄 Training Logistic Regression...")
    model.fit(resume_embeddings, df["Category_Encoded"])

    # Save Model
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("✅ Model training completed & saved.")

else:
    print("✅ Model already exists.")
