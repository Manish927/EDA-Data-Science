NEWS AND MEDIA

## E-news Platform News Categorization
Description

Categorize and tag news articles for an e-news platform to demonstrate improved content organization and enhanced user engagement.

# News Classification: From custom SLM to DistilBERT

A comprehensive comparison of News Categorization using custom Bi-Directional LSTMs (Small Language Models) and Fine-Tuned DistilBERT Transformers.

## 🚀 Project Overview
This project classifies news headlines into 10 distinct categories (Politics, Business, Wellness, etc.) using the Huffington Post News Category Dataset. The core objective was to benchmark a custom-built architecture against a state-of-the-art transformer model.

## 📊 Performance Comparison
| Model | Accuracy | F1-Score (Weighted) | Notes |
| :--- | :--- | :--- | :--- |
| **Custom BiLSTM** | 80.92% | ~0.79 | Fast, lightweight, but struggled with context. |
| **DistilBERT** | **85%+** | **0.86+** | Superior context handling and semantic understanding. |

## 🛠️ Tech Stack
* **Language:** Python
* **Frameworks:** PyTorch, Hugging Face Transformers
* **Models:** Bi-Directional LSTM, DistilBERT (base-uncased)
* **Visualization:** Seaborn, Matplotlib (Confusion Matrices)
* **Environment:** Google Colab (T4 GPU) / Local CUDA

## 🧠 Key Learnings
* **The "Glass Ceiling":** Custom SLMs are great for efficiency but struggle with nuanced categories like "Wellness" vs "Healthy Living."
* **Transfer Learning:** DistilBERT’s pre-trained knowledge of English syntax significantly boosts precision in overlapping categories.
* **Optimization:** Implemented Weight Decay, Dropout, and Learning Rate Schedulers to combat overfitting in deep architectures.

## 📈 Visualizing Confidence
The project includes a detailed Confusion Matrix analysis, identifying "semantic hotspots" where news categories overlap, providing insights into model decision-making.


