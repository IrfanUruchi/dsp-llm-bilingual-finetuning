# Fine-Tuning Lightweight Large Language Models for a Bilingual DSP Teaching Assistant

**Author:** Irfan Uruchi  
**Course:** Introduction to Data Science  
**Institution:** South East European University  

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Transformers-4.39+-violet?style=flat-square">
  <img src="https://img.shields.io/badge/PEFT-QLoRA-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Models-LLaMA%203.2--1B%20%7C%20Qwen%202.5--1.5B-green?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square">
</p>

---

<p align="center">
  <b>Bilingual DSP Teaching Assistant</b><br>
  Lightweight LLM Fine-Tuning • Verified Synthetic Data • English ↔ Albanian Support
</p>

---

# Project Overview

This repository contains the complete code, dataset samples, and training pipeline used to fine-tune a lightweight LLaMA 3.2–1B model into a bilingual (English–Albanian) Digital Signal Processing (DSP) teaching assistant.

The project demonstrates that:

- Small models (1B parameters) can perform well on DSP tasks.
- High-quality, verified synthetic data is often more important than raw model size.
- QLoRA enables full fine-tuning within 5–6GB of VRAM.
- A bilingual assistant is possible even for low-resource languages such as Albanian.

This repository is structured so that all experiments can be reproduced.

---

# Key Features

- **Lightweight fine-tuning with QLoRA**  
  Full model fine-tuning on 5–6GB of VRAM using 4-bit quantisation and LoRA adapters.

- **Bilingual DSP dataset (EN–SQ)**  
  Includes manually written and synthetic examples aligned across English and Albanian.

- **Synthetic problem generator**  
  Creates verified DSP questions covering FFT, aliasing, sampling, and related tasks.

- **Teacher–student refinement**  
  Larger LLaMA and Qwen models were used to refine clarity, translations, and consistency.

- **Evaluation pipeline**  
  Numeric verification scripts and manual scoring methods for conceptual correctness.

- **Cross-model comparison**  
  Benchmarks against multiple models ranging from 1B to 9B parameters.

---

# Results Summary

### Fine-Tuned vs. Base Model (LLaMA 3.2–1B)

- **Base model accuracy:** ~30–35%
- **Fine-tuned model accuracy:**
  - English: ~80%
  - Albanian: ~70%
  - Bilingual: ~68%

The fine-tuned 1B model performs reliably on core DSP tasks such as FFT bin calculation, aliasing detection, and sampling-related computations.

---

### Cross-Model Observations

The project also evaluated several models from 1.5B to 9B parameters. Notable findings:

- The fine-tuned 1B model approaches the performance of mid-range models (3B–7B).
- It outperforms Alpaca-7B on numerical DSP tasks due to better alignment and dataset quality.
- Data correctness and task-specific formatting had more impact than raw model size.

---

# Dataset Description

The dataset used for this project consists of bilingual (English–Albanian) DSP questions and answers.  
It was developed in several stages:

### 1. Manual Base Dataset
A small initial set of DSP questions was written manually, covering:
- FFT bin index calculations  
- aliasing and sampling scenarios  
- basic filter concepts  

Each item contains:
- an English prompt  
- an Albanian equivalent  
- a verified numeric answer  
- a short explanation  

### 2. Glossary for Terminology
A custom English–Albanian DSP glossary was created to:
- standardise translations,  
- ensure consistent use of technical terms,  
- avoid common Albanian grammar mistakes and missing diacritics.

### 3. Synthetic Dataset Expansion
A synthetic generator was implemented to produce additional examples.  
Each generated question:
- follows deterministic DSP formulas,  
- computes the expected answer,  
- and is checked again through verification scripts.

### 4. Teacher–Student Refinement
Larger LLaMA and Qwen models were used to refine:
- question clarity,  
- Albanian translations,  
- and explanation consistency.

Only items that passed all verification checks were included in training.

### Dataset Goals
The focus was on **quality over quantity**:
- clean formatting,  
- correct calculations,  
- consistent bilingual alignment,  
- and low noise.

This approach enabled a small 1B model to achieve high accuracy on DSP tasks.

---

# Training

The model was fine-tuned using **QLoRA**, which allows efficient training of large models on limited VRAM by combining:

- 4-bit quantisation for the base model  
- LoRA adapters for trainable parameters  
- Gradient accumulation to simulate larger batch sizes  

### Hardware Used
Training was performed on:
- GPU: NVIDIA RTX 3050 (8GB VRAM available, ~5–6GB used)
- RAM: 64GB
- OS: Windows 11 + Ubuntu 22.04 (WSL2)

### Software Stack
- Python 3.12  
- PyTorch + CUDA  
- Hugging Face Transformers  
- bitsandbytes (for 4-bit loading)  
- PEFT (LoRA / QLoRA)  
- accelerate  

### Training Steps
1. Load the LLaMA 3.2–1B model in 4-bit mode.  
2. Attach LoRA adapters to attention and feed-forward layers.  
3. Train on the bilingual DSP dataset for several epochs.  
4. Validate outputs using the numeric verification scripts.  
5. Save the resulting fine-tuned LoRA adapter and merged model.

The total VRAM usage stayed within **5–6GB**, making the process accessible for students and individual researchers.

---

# Evaluation

The model was evaluated across three different pipelines:

1. **English-only DSP questions**  
2. **Albanian-only DSP questions**  
3. **Mixed bilingual prompts** (English ↔ Albanian)

Each question was scored using a combination of:

### • Numeric correctness  
The output is parsed and checked using Python verification scripts.  
Tasks include:
- FFT bin index calculation  
- aliasing frequency computation  
- sampling relationships  

### • Reasoning clarity  
Short explanations are scored for:
- logical steps,
- formula correctness,
- consistency with DSP concepts.

### • Terminology accuracy (especially in Albanian)  
Checks include:
- proper use of DSP terms,
- correct diacritics (ë, ç),
- absence of unnecessary code-switching.

### • Output stability  
Evaluates whether the model avoids:
- hallucinated constants,
- incorrect equations,
- random formatting artifacts.

---

### Evaluation Highlights

- The **fine-tuned 1B model** showed stable behavior across all pipelines.  
- English performance was the strongest.  
- Albanian performance was lower but still reliable.  
- Mixed-language prompts were the most challenging but remained usable.  
- Numerically, the model avoided major mistakes thanks to the verified dataset.

All evaluation figures can be found in the `results/figures/` directory.

---

# Fine-Tuned Models

Two fine-tuned bilingual DSP models are available on HuggingFace:

---

### **LLaMA 3.2–1B DSP (English–Albanian)**  
Fine-tuned using QLoRA on the verified bilingual DSP dataset.

🔗 https://huggingface.co/Irfanuruchi/llama_3_2_1B_dsp-llm-bilingual

---

### **Qwen 2.5–1.5B DSP (English–Albanian)**  
Fine-tuned using the same pipeline to compare behaviour across architectures.

🔗 https://huggingface.co/Irfanuruchi/qwen2.5-1.5b-dsp-finetuned

---

Both models:
- support English, Albanian, and mixed-language prompts  
- handle basic DSP numerical reasoning  
- follow the same instruction template used during training  
- are lightweight enough for CPU or small-GPU inference

---

# Full Project Report

The complete academic paper for this project, submitted as part of the  
*Introduction to Data Science* course, is included in the repository:

Irfan_Uruchi_-_Fine-Tuning_Lightweight_Large_Language_Models_for_a_Bilingual_DSP_Teaching_Assistant_-_Introduction_to_Data_Science.pdf


The report contains:

- detailed methodology (six development stages)  
- dataset design and verification processes  
- fine-tuning setup and configuration  
- evaluation results across all tested models  
- discussion and future work  

It serves as the formal documentation for this project.

---

# Future Work

There are several directions in which this project can be extended:

### • Broader DSP Coverage
Include additional DSP topics such as:
- convolution,
- z-transform,
- filter design,
- time–frequency analysis,
- frequency response interpretation.

### • Improved Albanian Support
Enhance the linguistic quality of Albanian outputs by:
- incorporating a larger Albanian corpus,
- applying grammar-focused data augmentation,
- refining terminology consistency.

### • User Interface
Develop a simple web or desktop interface where students can:
- enter DSP questions,
- receive step-by-step solutions,
- choose English or Albanian output.

### • Additional Fine-Tuning Techniques
Test reinforcement-learning-based methods (RLHF, DPO) to improve:
- explanation clarity,
- reasoning structure,
- error recovery.

### • Model Comparison Expansion
Evaluate more lightweight and mid-size models to understand:
- where small models plateau,
- which architectures adapt best to DSP concepts.

---

# Acknowledgements

I would like to thank **Professor Nuhi Besimi** for his guidance throughout this project  
and for approving the use of this fine-tuning framework as part of the course.  
His feedback helped shape the structure and direction of the work.

---

# License

This project is open-source and released under the **MIT License**.

You are free to use, modify, and distribute the code as long as the original license terms are respected.

