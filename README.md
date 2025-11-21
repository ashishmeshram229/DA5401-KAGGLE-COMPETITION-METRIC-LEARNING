# DA5401 Data Challenge: Metric Learning for AI Evaluation

**Author:** Ashish Rajhans Meshram  
**Roll No:** DA25M016

## 📌 Overview
This repository contains the solution for the DA5401 End-Semester Data Challenge. The objective was to build a **Metric Learning model** to predict the "fitness score" (0-10) of a conversational AI's response based on a specific evaluation metric.

The challenge involved **Multimodal Learning**, requiring the fusion of:
1.  **Multilingual Text:** User prompts, system prompts, and responses (Tamil, Hindi, English, etc.).
2.  **Dense Embeddings:** Pre-computed 768-dimensional vectors representing abstract metric definitions.

## 🚀 Approach: Extreme Fine-Tuning

To handle the linguistic diversity and high-dimensional embeddings, I implemented a **Multimodal Transformer Architecture** rather than traditional statistical ML.

### Architecture Details
* **Text Encoder:** **XLM-RoBERTa-base** (Cross-lingual Language Model) to handle Indic languages and English effectively.
* **Fusion Layer:** Concatenation of the XLM-R `[CLS]` token output (Text Context) with the provided Metric Embeddings (Metric Intent).
* **Regression Head:** A custom Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout to predict the scalar score.

### Training Strategy
* **Optimizer:** AdamW with linear warmup and decay.
* **Optimization:** Gradient Accumulation (to simulate larger batch sizes) and Automatic Mixed Precision (AMP) for memory efficiency.
* **Loss Function:** Mean Squared Error (MSE).

## 📂 Files Description
* `multimodal_transformer_finetune.py`: The main training and prediction script.
* `train_data.json`: Training dataset with text pairs and scores.
* `metric_name_embeddings.npy`: Pre-computed embeddings for metric definitions.
* `metric_names.json`: List of evaluation metrics.

## 🛠️ Dependencies
To run the code, the following libraries are required:

```bash
pip install torch transformers pandas numpy scikit-learn
