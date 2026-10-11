# Skin Lesion Diagnosis and Report Generation through Vision-Language Models

This repository contains the codebase and experimental pipeline for adapting pretrained vision-language models (VLMs) to dermatoscopic image analysis. Rather than treating skin lesion diagnosis as a standard black-box classification task, this system pairs dermoscopic images with clinical metadata to generate structured, human-interpretable diagnostic reports alongside diagnostic reasoning.

---

## Overview

Traditional convolutional neural networks (e.g., ResNet, EfficientNet) achieve dermatologist-level accuracy but output only discrete class predictions without clinical justification. This project explores a multimodal decision-support framework that:
1. Couples dermoscopic images with structured clinical metadata (patient age, sex, lesion localization).
2. Employs a synthetic reporting pipeline to produce structured clinical reference notes.
3. Fine-tunes a pretrained LLaMA-based vision-language architecture using parameter-efficient fine-tuning (LoRA) and 4-bit quantization.
4. Evaluates generation fidelity (ROUGE-L), diagnostic classification accuracy, and visual attention alignment.

---

## Dataset & Preprocessing

The pipeline utilizes the **HAM10000** benchmark dataset (10,015 dermoscopic images across 7 diagnostic categories):
- **Classes**: Actinic keratoses and intraepithelial carcinoma (`akiec`), basal cell carcinoma (`bcc`), benign keratosis-like lesions (`bkl`), dermatofibroma (`df`), melanoma (`mel`), melanocytic nevi (`nv`), vascular lesions (`vasc`).
- **Data Cleaning**: Stripped records with missing demographic variables; standardized patient age to discrete integers.
- **Class Balancing**: Balanced the skewed dataset to a controlled sample range per diagnostic class (112–500 samples per class) to prevent majority-class collapse.
- **Synthetic Report Ground Truth**: Constructed structured medical note templates incorporating anatomical site, demographics, and distinctive morphological markers (e.g., vascular rosettes, arborizing telangiectasias, dark round structures).

---

## Model & Architecture

- **Architecture**: Pretrained LLaMA-based Vision-Language Model.
- **Fine-Tuning**: Low-Rank Adaptation (LoRA) on linear projection layers.
- **Quantization**: 4-bit precision (QLoRA) to allow execution within constrained GPU environments (Google Colab / single consumer GPU).
- **Objective**: Conditional text generation conditioned on `[Image + Patient Metadata]` inputs.

---

## Key Experimental Results

### Per-Category Report Quality (ROUGE-L)
Conditioning on patient metadata consistently boosted structural and linguistic similarity across all 7 diagnostic classes:

| Diagnosis | Image + Metadata | Image Only |
| :--- | :---: | :---: |
| **nv** | **0.28** | 0.22 |
| **vasc** | **0.25** | 0.20 |
| **bcc** | **0.27** | 0.18 |
| **bkl** | **0.24** | 0.21 |
| **df** | **0.23** | 0.19 |
| **mel** | **0.22** | 0.17 |
| **akiec** | **0.20** | 0.16 |

### Diagnostic Extraction Accuracy ($n = 13$ per class)
Diagnostic labels extracted via keyword matching from generated text outputs

| Diagnosis | Image + Metadata | Image Only |
| :--- | :---: | :---: |
| **nv** | **0.69** | 0.69 |
| **vasc** | 0.23 | 0.23 |
| **bcc** | **0.15** | 0.00 |
| **bkl** | 0.15 | **0.23** |
| **df** | 0.15 | **0.31** |
| **mel** | 0.08 | **0.23** |
| **akiec** | 0.00 | 0.00 |
