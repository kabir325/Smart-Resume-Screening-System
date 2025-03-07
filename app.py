import os
import sys
import pdfplumber
import joblib
import torch
import re
import pandas as pd
from transformers import BertTokenizer, BertModel
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextBrowser
from tqdm import tqdm

# Paths
MODEL_PATH = "models/resume_classifier_bert.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# Load BERT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_MODEL = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

# Text Cleaning Function
def clean_text(text):
    return re.sub(r'\W', ' ', str(text)).strip().lower()

# Convert Text to BERT Embeddings
def get_bert_embedding(text):
    tokens = TOKENIZER(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = BERT_MODEL(**tokens)
    return output.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# Ensure Model Exists
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("❌ Model not found. Run `train_model.py` first!")
    sys.exit(1)

# Load Trained Model
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " " if page.extract_text() else ""
    return clean_text(text)

# Predict Job Roles
def predict_resume(resume_text):
    resume_embedding = get_bert_embedding(resume_text).reshape(1, -1)
    predicted_probs = model.predict_proba(resume_embedding)[0]
    
    top_indices = predicted_probs.argsort()[-3:][::-1]
    return label_encoder.inverse_transform(top_indices)

# GUI App
class ResumeClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Resume Classifier")
        self.setGeometry(100, 100, 500, 300)
        
        layout = QVBoxLayout()
        self.label = QLabel("Upload your resume (PDF):")
        layout.addWidget(self.label)
        
        self.upload_button = QPushButton("Upload PDF")
        self.upload_button.clicked.connect(self.upload_pdf)
        layout.addWidget(self.upload_button)
        
        self.result_label = QLabel("Top matching job roles:")
        layout.addWidget(self.result_label)
        
        self.result_box = QTextBrowser()
        layout.addWidget(self.result_box)
        
        self.setLayout(layout)
    
    def upload_pdf(self):
        options = QFileDialog.Option.DontUseNativeDialog  # or QFileDialog.Option(0) for no options
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)", options=options)
        
        if file_path:
            self.result_box.setText("🔄 Processing resume...")
            resume_text = extract_text_from_pdf(file_path)
            job_matches = predict_resume(resume_text)
            self.result_box.setText("\n".join(job_matches))

# Run GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResumeClassifierApp()
    window.show()
    sys.exit(app.exec())
