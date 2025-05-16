from flask import Flask, request,redirect,url_for, render_template
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from flask import Flask, request, make_response
from itertools import combinations
from time import time
from collections import Counter
import operator
import math
import pickle
import openpyxl
from sklearn.linear_model import LogisticRegression
from flask import session
from utils import save_array_to_excel, clear_excel_data, fetch_and_merge_array__from_excel, fetch_values_from_excel, clear_variable_value, get_possible_symptoms, load_all_symptoms, save_appointment, predict_diseases_from_symptoms
from flask import request, render_template, session
from collections import Counter
import ast
warnings.simplefilter("ignore")


app=Flask(__name__,static_url_path='/static')
app.secret_key = 'super secret key'    

unselected_symptoms = []
predicted_diseases = []
predicted_symptoms = []
predicted_diseases_2 = []
symptoms_select_1 = []
symptoms_select_2 = []


@app.route("/",methods=["POST","GET"])
def index():
    all_symptoms = load_all_symptoms()
    save_array_to_excel('all_symptoms', all_symptoms)
    resp = make_response('Setting the cookie') 
    resp.set_cookie('GFG','ComputerScience Portal')
    GFG = request.cookies.get('GFG')
    
    return render_template("index.html")

@app.route("/about",methods=["POST","GET"])
def about():
    return render_template("about.html")

@app.route("/index",methods=["POST","GET"])
def index1():
    return render_template("index.html")

@app.route("/demo")
def demo():
    from utils import load_all_symptoms
    all_symptoms = load_all_symptoms()
    return render_template('demo.html', all_symptoms=all_symptoms)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    user_symptoms = request.form.getlist('symptoms[]')
    save_array_to_excel('user_symptoms_1', user_symptoms)
    symptoms_select_1 = user_symptoms
    results = predict_diseases_from_symptoms(user_symptoms)

    # Store only diseases in a list
    for _, prediction in results.items():
        predicted_symptoms.append(prediction)
 
    # Call the function to get possible symptoms from CSV
    possible_symptoms = get_possible_symptoms(predicted_symptoms)
    
    # Create a list of unselected symptoms (those that are in possible_symptoms but not in user_symptoms)
    select_2 = [symptom for symptom in possible_symptoms if symptom not in user_symptoms]
    unselected_symptoms = [
        symptom for symptom in possible_symptoms
        if symptom.strip("'").strip().lower() not in [s.strip().lower() for s in user_symptoms]
    ]

    unselected_symptoms = [symptom.replace("'", "").strip() for symptom in unselected_symptoms]
    select_2 = [symptom.replace("'", "").strip() for symptom in unselected_symptoms]

    return render_template(
        "predict.html",
        found_symptoms=enumerate(user_symptoms),
        another_symptoms=enumerate(unselected_symptoms),
        select_2=enumerate(select_2),
    )

@app.route("/next",methods=["POST","GET"])
def next():
    symptoms_select_2 = request.form.getlist('symptoms_select_2[]')
    if not symptoms_select_2:  # Checks if it's None or empty list
        clear_variable_value('symptoms_select_2')
    save_array_to_excel('user_symptoms_2', symptoms_select_2)
    
    final_symptoms=fetch_and_merge_array__from_excel()
    
    final_diseases = predict_diseases_from_symptoms(final_symptoms)
    
    # Count the occurrences of each disease
    disease_list = list(final_diseases.values())
    most_common_disease = max(disease_list, key=disease_list.count)

    # Wrap it in a list if you want to keep it as a "distinct_diseases" list
    distinct_diseases = [most_common_disease]
    distinct_diseases = list(set(str(disease) for disease in final_diseases.values()))
    save_array_to_excel('final_diseases', distinct_diseases)

    return render_template("next.html",diseases=distinct_diseases)


@app.route("/treatment",methods=["POST","GET"])
def treatment():
    treat_dis = request.form.get('treat_dis')
    workbook = openpyxl.load_workbook("Dataset\medical_remedy_map.xlsx")
    worksheet = workbook['Sheet1']
    for row in worksheet.iter_rows(values_only=True):
        if treat_dis in row:
            stri = ''.join(row[1:])
            ans=stri.split(',') 
    
    return render_template("treatment.html",ans=ans)


@app.route("/appointment", methods=["POST"])
def appointment():
    # Get form data
    full_name = request.form.get("full_name")
    email = request.form.get("email")
    appointment_date = request.form.get("appointment_date")
    department = request.form.get("department")
    phone_number = request.form.get("phone_number")
    message = request.form.get("message")

    # Save to Excel
    save_appointment(full_name, email, appointment_date, department, phone_number, message)

    # save_appointment(
    #     full_name="Anna Kowalska",
    #     email="anna.kowalska@mail.pl",
    #     appointment_date="2025-05-18",
    #     department="Dental",
    #     phone_number="+48 987 654 321",
    #     message="Need to schedule a teeth cleaning appointment."
    # )

    return redirect("/")

if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0", port=5000,threaded=True)