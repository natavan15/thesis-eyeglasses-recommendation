# Thesis Eyeglasses Recommendation Project

This repository contains the source code, prompts, result files, and project structure used for the master thesis:

**Research on Automatic Generation of Eyeglasses Design Recommendations Based on Facial Features**

The thesis investigates AI-based eyewear recommendation using face-shape classification and reasoning-oriented recommendation methods. The study compares a traditional CNN-based pipeline with cloud-based and local multimodal Vision-Language Model pipelines.

## Project Overview

The project evaluates three main approaches:

1. **CNN-based pipeline**
   - Model: InceptionV4
   - Task: Face-shape classification
   - Output: Predicted face-shape class
   - Recommendation: Rule-based eyewear suggestion using selected frame examples

2. **Cloud-based VLM pipeline**
   - Models: GPT-4o / GPT-4o-mini
   - Task: Face-shape classification using image prompts
   - Output: Structured JSON prediction with confidence and reasoning

3. **Local VLM pipeline**
   - Model: Gemma 4
   - Task: Local multimodal face-shape classification
   - Output: Predicted face-shape class and reasoning-oriented recommendation support

## Face-Shape Classes

All models are evaluated using five face-shape categories:

- heart
- oblong
- oval
- round
- square

## Repository Structure

```text
thesis-eyeglasses-recommendation/
│
├── computer_vision_pipeline/
│   ├── results/
│   ├── src/
│   ├── prepare_datasets.py
│   └── test_dataset_loader.py
│
├── llm_pipeline/
│   ├── results/
│   └── src/
│
├── openai_vision_pipeline/
│   ├── prompts/
│   ├── results/
│   └── src/
│
├── requirements.txt
├── .gitignore
└── README.md
