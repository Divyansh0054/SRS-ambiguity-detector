# SRS Ambiguity Detection System

##  Project Overview
This project focuses on detecting ambiguity in Software Requirement Specification (SRS) documents using a hybrid approach that combines Natural Language Processing (NLP) and Machine Learning (ML).

Ambiguity in requirements can lead to miscommunication, incorrect implementation, and project delays. This system helps identify unclear requirements and suggests improvements.

---

## Features

- Detects ambiguous words and phrases
- Highlights ambiguous terms in sentences
- Provides reasons and suggestions for improvement
- Uses Machine Learning for classification
- Hybrid decision system (Rule-based + ML)
- Supports:
  - Text input
  - PDF upload
  - DOCX upload
- Generates downloadable analysis report

---

##  Approach

###  Rule-Based NLP
- Detects:
  - Modal verbs (should, may, could)
  - Subjective words (fast, efficient, user-friendly)
  - Vague phrases
- Provides explanation and suggestions

###  Machine Learning
- TF-IDF for feature extraction
- Logistic Regression for classification
- Outputs:
  - Prediction (Ambiguous / Clear)
  - Confidence score

###  Hybrid System
- Combines rule-based and ML results
- Final decision:
  - Ambiguous
  - Clear
  - Needs Review

---

## 📂 Project Structure
── app.py # Main Streamlit application
├── detector/
│ └── ambiguity_detector.py # Rule-based NLP logic
├── ambiguity_model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── requirements.csv # Dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation


## Contributors
- Divyansh Sharma
- Vansika Jain
