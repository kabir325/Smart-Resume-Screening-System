# **📄 Smart Resume Screening System 🚀**  

🎯 **Automatically match resumes with the best job descriptions using NLP & Machine Learning (BERT).**  

## **🔹 Project Overview**
This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify resumes into different job categories.  
🔹 **Input**: Resume (PDF)  
🔹 **Output**: Top 3 matching job roles  

✅ **Pretrained BERT for text embeddings**  
✅ **Machine Learning model for classification**  
✅ **Interactive GUI for resume upload**  
✅ **Real-time training and processing updates**  

---

## **📌 Features**
✔️ **AI-powered Resume Matching** – Uses BERT embeddings to predict job roles  
✔️ **Supports PDF Resumes** – Extracts text from resumes automatically  
✔️ **User-Friendly GUI** – Upload and get predictions instantly  
✔️ **Fast & Efficient** – Optimized using `Logistic Regression`  
✔️ **Auto Train & Setup** – Trains models if not found  

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/kabir325/Smart-Resume-Screening-System.git
cd Smart-Resume-Screening-System
```

**2️⃣ Create & Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

**3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

**4️⃣ Download Dataset**

*Download the Kaggle dataset from the following link*

https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset?resource=download


**5️⃣ Train the Model**

```bash
python train_model.py
```

**6️⃣ Run the Application**

```bash
python app.py
```

**🖥️ Usage**
1️⃣ Open the application 2️⃣ Click **"Upload PDF"** and select your resume 3️⃣ Get **top 3 job recommendations** instantly
**📂 Project Structure**

```
📂 resume-classifier
│-- 📂 dataset/            # Stores downloaded dataset
│-- 📂 models/             # Stores trained models
│-- download_dataset.py    # Downloads dataset from Kaggle
│-- train_model.py         # Trains the BERT-based model
│-- app.py                 # Runs GUI for resume upload & classification
│-- requirements.txt       # Project dependencies
│-- README.md              # Documentation
```

**📊 Model Details**
🔹 **Embedding Model**: `BERT (bert-base-uncased)` 🔹 **Classifier**: `Logistic Regression` 🔹 **Evaluation**: `Scikit-Learn Accuracy, Precision, Recall` 🔹 **Text Processing**: `pdfplumber for PDF parsing`
**📷 Screenshots**
**Upload Resume** **Predicted Job Roles**
**📌 Dependencies**
* `torch`
* `transformers`
* `pdfplumber`
* `joblib`
* `scikit-learn`
* `pandas`
* `numpy`
* `nltk`
* `PyQt6`
* `kaggle`
* `tqdm`
🔹 **Install all with:**

```bash
pip install -r requirements.txt
```

**🙌 Contributing**
🚀 Contributions are welcome!
* Fork the repo
* Create a new branch
* Commit your changes
* Open a Pull Request
**📜 License**
This project is **open-source** under the MIT License.
**📞 Contact**
📧 **Email:** yourname@email.com 🐦 **Twitter:** @yourhandle 👨‍💻 **GitHub:** yourusername
🎯 **Star 🌟 this repo if you found it useful!** 🚀
