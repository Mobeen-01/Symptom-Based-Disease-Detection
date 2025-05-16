# MediCURE Disease Prediction Based on Symptoms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-1.1.2-lightgrey.svg?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Medical%20Data-orange.svg?style=flat)](https://www.kaggle.com)
[![ML Models](https://img.shields.io/badge/ML%20Models-LogReg%20%7C%20SVM%20%7C%20ANN%20%7C%20XGBoost-brightgreen.svg)]()

---

## Introduction

**MediCURE** is an AI-powered web application designed to predict diseases based on user-reported symptoms, providing confidence scores and recommending home remedies for treatment. The app is built with **Flask** and leverages multiple machine learning models trained on a large medical dataset.



> 🚨 **Note:** This project is intended for informational purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical concerns.

---

## Features

- ✅ **Symptom-Based Disease Prediction:** Uses machine learning models including Logistic Regression, Support Vector Machine (SVM), Multilayer Perceptron (MLP), Artificial Neural Network (ANN), Naive Bayes, Decision Tree, Random Forest, and XGBoost.
- ✅ **Confidence Scoring:** Provides probabilistic confidence scores with each prediction.
- ✅ **Home Remedies Recommendation:** Suggests natural home remedies tailored to the predicted diseases.
- ✅ **Comprehensive Dataset:** Trained on a Kaggle dataset containing 8,835 unique rows, 490 distinct symptoms, and 261 distinct diseases.
- ✅ **Comparative Model Analysis:** Evaluates various ML algorithms to identify the best-performing model.
- ✅ **User-Friendly Web Interface:** Easy-to-use Flask web application.
- ✅ **Data Augmentation:** Enhanced dataset via symptom combination expansion.
- ✅ **Informational Use:** Designed for awareness, **not** as medical advice or diagnosis.

---

## Demo Video

> Click below to watch the demonstration of the application in action:

[![MediCURE Demo](https://img.youtube.com/vi/79ac30cd-e462-4601-85eb-774a14658ab7/0.jpg)]([https://user-images.githubusercontent.com/107244393/235594013-79ac30cd-e462-4601-85eb-774a14658ab7.mp4](https://github.com/Mobeen-01/Symptom-Based-Disease-Detection/blob/main/Demo/Demo.mp4))



---

## Dataset

- Dataset source: Kaggle (secondary data)
- Size: 8,835 unique rows, 490 distinct symptoms, 261 distinct diseases
- Format: Symptoms as attributes, diseases as labels
- Data augmentation by combining symptom sets to increase dataset diversity
- Additional treatment dataset: `cure_minor.xlsx`

---

## Technologies Used

- **Backend & Web:** Python, Flask
- **Machine Learning Models:** Logistic Regression, SVM, MLP, ANN, Naive Bayes, Decision Tree, Random Forest, XGBoost
- **Data Handling:** Pandas, NumPy
- **Version Control:** Git, GitHub

---


## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MediCURE-Disease-Prediction.git
cd MediCURE-Disease-Prediction
```

### 2.  Set Up Environment and Install Dependencies

Open your terminal and run the following commands:Open your terminal and run the following commands:

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate

pip install flask scikit-learn xgboost pandas numpy requests 
pip install nltk
pip install beautifulsoup4
pip install openpyxl
pip install googlesearch-python
pip install neural_structured_learning
pip install werkzeug==2.3.7
pip install matplotlib
pip install tensorflow
```

### 3. Run the application

Start the Flask web app by running:

```bash
python app.py
```

### 🔍 Usage

- Open your browser and go to `http://127.0.0.1:5000`
- Enter symptoms you are experiencing in the web interface
- Submit to get predicted diseases with confidence scores
- View recommended home remedies for the predicted disease

### 📼 Watch the Guide

After unzipping the project folder, **please watch `Guide.mp4`** in the root directory for detailed instructions on:

- Understanding the project structure  
- Installing the required libraries  
- Running the application smoothly

The video file is located at the root of the repository:

```bash
Guide.mp4

```
## 📹 Demo

A walkthrough video is included in the repo: `Demo/Demo.mp4`



## 📜 License

This project is licensed under the MIT License.


