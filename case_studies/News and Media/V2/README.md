# 🚀 DistilBERT from Scratch: Building an Efficient NLP Pipeline (EDA → BERT → Distillation)

## 📌 Project Overview

This project demonstrates an **end-to-end NLP pipeline** for text classification, progressing from **Exploratory Data Analysis (EDA)** to **fine-tuning a BERT model**, and finally building a **DistilBERT-like Small Language Model (SLM)** using **knowledge distillation**.

🎯 **Goal:**
Improve performance from traditional models (~80%) to **Transformer-based models (85–90%+)**, while reducing model size and improving inference speed.

---

## 🧠 Key Highlights

* ✅ Built a **BERT-based Teacher Model**
* ✅ Designed a **lightweight Student Model (DistilBERT-style)**
* ✅ Applied **Knowledge Distillation (KL + CE Loss)**
* ✅ Achieved **comparable accuracy with reduced size**
* ✅ Added **advanced evaluation & interpretability**

  * F1 Score comparison
  * ROC curves
  * Confusion matrices
  * Attention visualization
  * Inference speed comparison
  * Model size comparison

---

## 🏗️ Project Pipeline

```
01_eda_news_classification.ipynb
        ↓
02_bert_finetuning.ipynb   (Teacher Model)
        ↓
03_distillation_student_model.ipynb   (Student Model)
```

---

## 📊 Dataset

* 📰 News Classification Dataset (AG News / Custom dataset)
* Multi-class classification problem
* Balanced dataset with clean text distribution

---

## 🔍 Step 1: Exploratory Data Analysis (EDA)

Key insights:

* Average text length ≈ **37 words**
* Most samples fall between **20–50 words**
* Minimal preprocessing required for BERT

📌 Decisions:

* Max sequence length = **64**
* Light text cleaning (no stemming/lemmatization)

---

## 🧠 Step 2: BERT Fine-Tuning (Teacher Model)

* Model: `bert-base-uncased`
* Training:

  * Epochs: 3
  * Batch size: 16
  * Learning rate: 2e-5

📈 Output:

* High accuracy baseline
* Saved as: `bert_teacher_model/`

---

## ⚡ Step 3: Knowledge Distillation (Student Model)

### 🎯 Objective

Train a smaller model to mimic the teacher.

### 🔥 Loss Function

```
Loss = α * CrossEntropy + (1 - α) * KL Divergence
```

### 🧱 Student Model

* DistilBERT-style architecture
* Reduced layers (6 vs 12 in BERT)

📉 Result:

* ~40–60% smaller model
* Faster inference
* Minimal accuracy drop

---

## 📊 Performance Comparison

| Metric     | Teacher (BERT) | Student (Distilled) |
| ---------- | -------------- | ------------------- |
| Accuracy   | High           | Slightly lower      |
| F1 Score   | High           | Comparable          |
| Speed      | Slower         | Faster ⚡            |
| Model Size | Large          | Smaller 📉          |

---

## 📈 Visualizations

### 🔹 Training Loss

* Shows convergence of student model

### 🔹 Accuracy & F1 Comparison

* Demonstrates minimal performance drop

### 🔹 ROC Curve

* Multi-class ROC using One-vs-Rest

### 🔹 Confusion Matrix

* Highlights class-wise performance

### 🔹 Inference Speed

* Student model significantly faster

### 🔹 Model Size

* Reduced memory footprint

---

## 🧠 Attention Visualization (Interpretability)

We visualized attention weights to understand model behavior:

* Focus on key tokens like:

  * *stock, markets, earnings*
* Strong attention on `[SEP]` token (sequence representation)
* Demonstrates meaningful contextual learning

📌 Insight:

> Transformer models learn semantic importance rather than relying on surface-level patterns.

---

## ⚙️ Tech Stack

* Python 🐍
* PyTorch 🔥
* Hugging Face Transformers 🤗
* Datasets & Evaluate
* Scikit-learn
* Matplotlib

---

## 🚀 Key Learnings

* Difference between **traditional ML vs Transformer models**
* Importance of **tokenization & sequence length**
* How **knowledge distillation compresses models**
* Trade-offs between:

  * Accuracy vs Speed
  * Size vs Performance
* Understanding **attention mechanisms**

---

## 💼 Why This Project Matters

This project demonstrates:

* End-to-end ML pipeline design
* Deep understanding of Transformer architecture
* Practical model optimization (distillation)
* Real-world trade-off analysis
* Strong debugging & experimentation skills

---

## 🔥 Future Improvements

* Add **attention distillation**
* Experiment with **TinyBERT / MobileBERT**
* Deploy model using **FastAPI / Streamlit**
* Optimize with **quantization**

---

## 📌 How to Run

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. EDA
2. BERT Training
3. Distillation

---

## 👨‍💻 Author

**Manish Srivastava**

---
