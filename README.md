# Generative AI Text Model from Scratch (Python)

This project implements a **basic Generative AI text model from scratch** using Python and PyTorch.  
The model learns **character-level language patterns** from text data and generates new text using an **autoregressive approach**, without relying on any pre-trained APIs or large language models.

---

## 📌 Project Objective

The goal of this project is to **understand how Generative AI models work internally**, including:

- How text is converted into numerical form
- How neural networks learn sequence patterns
- How next-token (character) prediction enables text generation
- How autoregressive generation creates the illusion of language understanding

This project focuses on **learning fundamentals**, not building a production-grade chatbot.

---

## 🧠 How the Model Works (High-Level)

1. **Text Preprocessing**
   - Raw text is loaded and cleaned
   - Converted to lowercase
   - Normalized to remove hidden or non-ASCII characters

2. **Tokenization (From Scratch)**
   - Character-level tokenization
   - Each unique character is mapped to a numeric index
   - Reverse mapping is used to convert predictions back to text

3. **Sequence Creation**
   - Fixed-length character sequences are created using a sliding window
   - Each sequence is paired with the next character as the target
   - This enables supervised next-character prediction

4. **Neural Network Model**
   - Embedding layer converts character indices into dense vectors
   - LSTM processes character sequences and captures context
   - Fully connected layer predicts the next character

5. **Training**
   - Trained using CrossEntropyLoss
   - Optimized with Adam optimizer
   - Learns statistical character patterns from the dataset

6. **Autoregressive Text Generation**
   - Given an initial prompt, the model predicts one character at a time
   - Each prediction is appended and reused as input for the next step
   - This creates continuous text generation

---

## ⚙️ Tech Stack

- **Python**
- **PyTorch**
- **uv** (for dependency and environment management)

---

