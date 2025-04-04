# Financial Sentiment Analysis: Naive Bayes vs. BERT

## 📌 Overview
This project explores **sentiment analysis of financial reports** using **Natural Language Processing (NLP)**. We compare the performance of:
- **Naive Bayes** (Traditional Statistical Model)
- **BERT** (Deep Learning Transformer Model)

The dataset used is the **Financial PhraseBank**, which contains sentiment-labeled financial news statements. The goal is to evaluate how well statistical models perform against transformer-based models in classifying sentiment (Positive, Neutral, Negative).

---

## 📊 Key Findings
- **Naive Bayes Accuracy**: `78.59%`
- **BERT Accuracy**: `96.69%`
- **BERT significantly outperforms Naive Bayes**, demonstrating its ability to understand complex financial sentiment.
- **Confusion Matrices** show that Naive Bayes struggles with negative sentiment, while BERT classifies all categories well.

---

## 📂 Dataset
**Financial PhraseBank** ([Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank))
- Contains ~4,840 financial news sentences labeled as **Positive, Neutral, or Negative**.
- Annotations are available at different agreement levels: `50%, 66%, 75%, 100%`.

Files included:
- `Sentences_50Agree.txt`
- `Sentences_66Agree.txt`
- `Sentences_75Agree.txt`
- `Sentences_AllAgree.txt`

---

## 🚀 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your_github_username/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```
### **2️⃣Download Pretrained BERT Model**
```bash
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
```
---

## 🛠 Model Implementation

### **Naive Bayes (Baseline Model)**
- **Feature Engineering**: Uses **TF-IDF Vectorization**.
- **Model**: `Multinomial Naive Bayes`
- **Performance**:
  - `Accuracy`: `78.59%`
  - `Precision`: `83.27%`
  - `Recall`: `57.90%`
  - `F1-Score`: `57.67%`

### **BERT (Transformer Model)**
- **Fine-tuned on Financial Sentiment Data**.
- **Model**: `BERT-base-uncased`
- **Performance**:
  - `Accuracy`: `96.69%`
  - `Precision`: `95.38%`
  - `Recall`: `95.53%`
  - `F1-Score`: `95.45%`

### **Confusion Matrix**
📉 **Naive Bayes struggles with negative sentiment classification.**
📈 **BERT accurately classifies all sentiment categories.**

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|------------|---------|-----------|
| **Naive Bayes** | `78.59%` | `83.27%` | `57.90%` | `57.67%` |
| **BERT** | `96.69%` | `95.38%` | `95.53%` | `95.45%` |

---

## 🔄 Model Training & Saving
**Naive Bayes Model**
```python
import joblib
joblib.dump(nb_classifier, 'models/naive_bayes_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
```

**BERT Model**
```python
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_tokenizer")
```

---

## 📊 Results & Comparison

| Metric | Naïve Bayes | BERT |
|-------------|------------|--------|
| **Accuracy** | 78.59% | 96.69% |
| **Precision** | 83.27% | 95.38% |
| **Recall** | 57.90% | 95.53% |
| **F1-Score** | 57.67% | 95.45% |

📢 **Conclusion**: BERT significantly improves financial sentiment classification by understanding contextual relationships in financial text.

---

## 📜 Future Improvements
🔹 Incorporate **FinancialBERT**, a transformer model fine-tuned for financial text.  
🔹 Try **ensemble models**, combining Naive Bayes and BERT predictions.  
🔹 Experiment with **new financial datasets** from EDGAR, Kaggle, or Bloomberg.  

---

## 📎 References
- Financial PhraseBank Dataset: [Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank)
- BERT: [Original Paper](https://arxiv.org/abs/1810.04805)
- Sentiment Analysis in Finance: [ResearchGate](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining)

---

## 💡 Contributing
Pull requests are welcome! Please open an issue first to discuss any changes.

---

## 👨‍💻 Author
**Kelvin Musodza**  
🔗 LinkedIn: [linkedin.com/in/kelvinmusodza](https://linkedin.com/in/kelvinmusodza)  
🐙 GitHub: [KELVI23](https://github.com/KELVI23)  

---

## ⭐ Support
If you like this project, **give it a star** ⭐ on GitHub!


---
