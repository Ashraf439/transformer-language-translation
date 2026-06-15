# English-to-French Neural Machine Translation Using an Enhanced Transformer Architecture

## Project Overview

This project presents the design and implementation of a neural machine translation system for English-to-French translation using a custom-built Transformer architecture developed entirely in PyTorch.

While inspired by the transformer-based translation approach described by Adrian Tam in the Machine Learning Mastery article, the original tutorial implementation was extensively redesigned and re-engineered into a modular, scalable, and maintainable software project. The objective was not only to reproduce translation results but also to apply modern Transformer improvements and software engineering best practices commonly used in production machine learning systems.

The resulting system provides a complete end-to-end translation pipeline, including dataset processing, Byte Pair Encoding (BPE) tokenization, model training, inference, and evaluation.

---

## Objectives

The primary objectives of this project are:

* Design and implement a Transformer-based Neural Machine Translation (NMT) system from scratch using PyTorch.
* Improve upon the baseline tutorial architecture through modern Transformer enhancements.
* Develop a modular and maintainable codebase suitable for experimentation and future research.
* Train and evaluate the model on a large-scale English-French parallel corpus.
* Provide an interactive inference interface for real-time translation.

---

## System Architecture

The proposed model follows an Encoder-Decoder Transformer architecture with several improvements over the original Transformer design.

### Architectural Features

| Component               | Implementation                      |
| ----------------------- | ----------------------------------- |
| Positional Encoding     | Rotary Positional Embeddings (RoPE) |
| Attention Mechanism     | Grouped Query Attention (GQA)       |
| Feed Forward Network    | SwiGLU Activation                   |
| Normalization           | RMSNorm (Pre-Norm Architecture)     |
| Encoder Layers          | 4                                   |
| Decoder Layers          | 4                                   |
| Hidden Dimension        | 128                                 |
| Vocabulary Size         | 8,000 tokens per language           |
| Maximum Sequence Length | 768 tokens                          |

### Encoder

The encoder consists of four stacked layers, each containing:

* RMSNorm
* Multi-Head Self-Attention with RoPE
* Residual Connections
* SwiGLU Feed-Forward Network

### Decoder

The decoder contains four layers with:

* Masked Self-Attention
* Encoder-Decoder Cross-Attention
* SwiGLU Feed-Forward Network
* RMSNorm and Residual Connections

---

## Key Contributions and Improvements

Compared to the original tutorial implementation, several architectural and software engineering improvements were introduced.

### Software Engineering Enhancements

* Complete modularization of the codebase.
* Separation of configuration, data processing, model definition, training, and inference.
* Centralized configuration management through a dedicated configuration module.
* Reusable inference pipeline through standalone decoding utilities.
* Platform-independent data loading implementation.

### Transformer Improvements

* Integration of Rotary Positional Embeddings (RoPE) instead of fixed sinusoidal positional encoding.
* Implementation of Grouped Query Attention (GQA) to improve computational efficiency.
* Adoption of SwiGLU feed-forward blocks for enhanced representation learning.
* RMSNorm-based pre-normalization architecture for improved training stability.
* Correct handling of cross-attention positional information.
* Optimized key-value projection dimensions for Grouped Query Attention.

### Maintainability Improvements

* Explicit encoder and decoder interfaces.
* Testable and reusable decoding functions.
* Elimination of global tokenizer dependencies.
* Clear separation between model logic and training logic.

---

## Dataset

The model is trained using the English-French sentence pair dataset provided by the Anki translation corpus.

### Dataset Characteristics

* Approximately 190,000 parallel sentence pairs.
* Automatic dataset download and preprocessing.
* Unicode NFKC normalization.
* Lowercase text standardization.
* Special start and end tokens for target sequences.
* Custom BPE tokenizers trained independently for English and French.

---

## Project Structure

```text
machine-translation/
├── config.py
├── utils.py
├── tokenizer.py
├── dataset.py
├── train.py
├── inference.py
├── main.py
└── model/
    ├── attention.py
    ├── feedforward.py
    ├── layers.py
    └── transformer.py
```

This structure promotes modularity, readability, maintainability, and ease of future expansion.

---

## Training Configuration

| Parameter         | Value |
| ----------------- | ----- |
| Batch Size        | 64    |
| Epochs            | 60    |
| Learning Rate     | 0.005 |
| Warmup Steps      | 1000  |
| Gradient Clipping | 5.0   |

### Learning Rate Strategy

The model employs:

1. Linear warmup during the initial training phase.
2. Cosine learning rate decay after warmup completion.

This approach improves optimization stability and convergence performance.

---

## Experimental Results

The trained model demonstrates strong translation quality on unseen English sentences.

### Example Outputs

**Input:**
Are there any bananas?

**Reference:**
Y a-t-il des bananes ?

**Prediction:**
Y a-t-il des bananes ?

---

**Input:**
I miss my parents.

**Reference:**
Mes parents me manquent.

**Prediction:**
Mes parents me manquent.

---

**Input:**
Turn left at the second traffic light.

**Reference:**
Tourne au second feu à gauche !

**Prediction:**
Tournez au deuxième feu à gauche !

---

## Technologies Used

* Python 3.10+
* PyTorch
* Hugging Face Tokenizers
* Requests
* TQDM

---

## Future Enhancements

Potential extensions of this work include:

* Beam Search Decoding
* Mixed Precision Training (FP16)
* Flash Attention Integration
* Larger Transformer Configurations
* BLEU Score Evaluation Framework
* Model Checkpoint Averaging
* ONNX Export and Deployment
* Web-Based Translation Interface

---

## Conclusion

This project demonstrates the practical implementation of a modern Transformer-based Neural Machine Translation system using PyTorch. Beyond reproducing a tutorial, the work focuses on architectural improvements, modular software design, and engineering best practices expected in real-world machine learning projects. The resulting system serves as a strong foundation for further research and deployment-oriented NLP applications.
