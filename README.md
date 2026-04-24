# Multi-Modal Document Classification

This repository implements a Late-Fusion Multimodal Architecture for document image classification. By leveraging both high-frequency visual features (CNN/ViT) and semantic textual features (OCR + Transformers), this approach achieves superior performance on noisy and structurally complex document datasets like Tobacco3482.

## Architecture

The model follows a two-stream "Late Fusion" strategy:

1. **Vision Branch**: Extracts spatial and structural features from document images using an EfficientNet or Vision Transformer (ViT) backbone.

2. **Text Branch**: Processes OCR-extracted text through a RoBERTa or BERT encoder to capture semantic context.

3. **Fusion Head**: Concatenates the feature vectors from both modalities, followed by an MLP with Dropout to perform final classification.

## Tech Stack

* **Python**: Version 3.12.
* **Environment**: uv for high-performance dependency management.
* **Framework**: PyTorch Lightning for modular and scalable model training.
* **Data**: Hugging Face Datasets for efficient metadata management and streaming.
* **Augmentations**: Using Albumentations library.
* **Experiment Tracking**: MLflow to track metrics, hyperparameters, and model versions.
* **Deployment**: ONNX-Runtime for optimized CPU/GPU inference.

## Evaluation Strategy
* **Dataset**: Tobacco3482 (scanned historical documents).
* **Primary Metric**: Macro-F1 Score (to handle significant class imbalance).
* **Ablation** Study: We train individual Image-only and Text-only models to quantify the accuracy lift provided by the multimodal fusion.
* **Serving**: Post-evaluation, the full graph is exported to ONNX for production-ready application hosting.

## Repository Structure
`src/`: Core logic including the LightningModule and LightningDataModule.
`data/`: Processed metadata.jsonl and stratified splits.
`deployment/`: ONNX inference scripts and FastAPI serving logic.
`notebooks/`: Exploration, EDA, and training runs on Google Colab.

## Getting Started

To setup the project run `scripts/project_setup.sh`.
This essentially creates empty untracked directories as well as setup the python environment.

From the project root, run the following commands using a linux bash [in windows, MYSY or git Bash should do the trick].

```shell
chmod +x scripts/project_setup.sh
bash scripts/project_setup.sh
```