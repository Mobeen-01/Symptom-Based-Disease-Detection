import warnings
from decimal import Decimal
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import ast
warnings.simplefilter("ignore")
import pandas as pd
from flask import session
import json
from flask import make_response
from flask import Flask, request, make_response
import pandas as pd
import os

EXCEL_FILE_PATH = 'user_data.xlsx'

# Load dataset once globally
df_comb = pd.read_csv("Dataset/dataset.csv")
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]
all_symptoms = list(X.columns)

# Define model paths
model_paths = {
    "Multinomial Naive Bayes": 'saved_models/mnb_model.pkl',
    "Random Forest": 'saved_models/rf_model.pkl',
    "K-Nearest Neighbors": 'saved_models/knn_model.pkl',
    "Logistic Regression": 'saved_models/lr_model.pkl',
    "Support Vector Machine": 'saved_models/svm_model.pkl',
    "Decision Tree": 'saved_models/dt_model.pkl',
    "Multilayer Perceptron": 'saved_models/mlp_model.pkl'
}

def predict_diseases_from_symptoms(symptoms: list) -> dict:
    """
    Predicts diseases using multiple trained models based on input symptoms.

    Args:
        symptoms (list): A list of symptom strings (e.g., ['fever', 'headache'])

    Returns:
        dict: A dictionary with model names as keys and predicted disease as values.
    """
    # Create input vector
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    
    predictions = {}
    for name, path in model_paths.items():
        model = joblib.load(path)
        prediction = model.predict([input_vector])[0]
        predictions[name] = prediction
    
    return predictions


def save_array_to_excel(variable_name, data_list):
    """
    Saves or updates the variable_name and its associated data_list in an Excel file.
    If the variable_name already exists, its value is updated; otherwise, a new row is added.
    """
    # Convert list to string as one entry
    new_data = str(data_list)

    # Check if file exists and is not empty
    if os.path.exists(EXCEL_FILE_PATH) and os.path.getsize(EXCEL_FILE_PATH) > 0:
        try:
            # Read the existing Excel file
            existing_df = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')

            # Check if the variable_name exists in the 'Variable Name' column
            if variable_name in existing_df['Variable Name'].values:
                # Update the existing value in the 'Value' column for the given variable_name
                existing_df.loc[existing_df['Variable Name'] == variable_name, 'Value'] = new_data
            else:
                # If variable_name doesn't exist, add it as a new row
                new_df = pd.DataFrame({
                    'Variable Name': [variable_name],
                    'Value': [new_data]
                })
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading or updating Excel: {e}")
            return
    else:
        # If the file doesn't exist or is empty, create a new DataFrame
        existing_df = pd.DataFrame({
            'Variable Name': [variable_name],
            'Value': [new_data]
        })

    # Save the updated DataFrame back to Excel
    existing_df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')
    print(f"Saved to {EXCEL_FILE_PATH}")

def clear_excel_data():
    """
    Clears all rows from the Excel file while preserving the column headers.

    This is useful when you want to reset the data but keep the file structure intact
    for future use (e.g., keeping columns like 'Variable Name' and 'Value').

    If the file doesn't exist, it logs a message instead of crashing.
    """
    if os.path.exists(EXCEL_FILE_PATH):
        try:
            # Read existing Excel file with headers
            df = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')
            
            # Create an empty DataFrame with the same headers
            empty_df = pd.DataFrame(columns=df.columns)
            
            # Write back to the file, overwriting it
            empty_df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')
            
            print(f"✅ Cleared all data in: {EXCEL_FILE_PATH} (headers kept)")
        except Exception as e:
            print(f"❌ Error clearing Excel file: {e}")
    else:
        print(f"⚠️ File not found: {EXCEL_FILE_PATH}")

def clear_variable_value(variable_name):
    """
    Clears the value in the second column for the row where the first column matches `variable_name`.

    Replaces the value with [] while keeping other data and headers intact.
    """
    if os.path.exists(EXCEL_FILE_PATH):
        try:
            # Read the Excel file
            df = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')

            # Check if the first column contains the variable_name
            if variable_name in df.iloc[:, 0].values:
                # Find the row index
                index_to_update = df[df.iloc[:, 0] == variable_name].index

                # Update the value in the second column to '[]'
                df.iloc[index_to_update, 1] = '[]'

                # Save the updated DataFrame back to the Excel file
                df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

                print(f"✅ Updated value for '{variable_name}' to [] in: {EXCEL_FILE_PATH}")
            else:
                print(f"⚠️ Variable '{variable_name}' not found in first column.")

        except Exception as e:
            print(f"❌ Error updating Excel file: {e}")
    else:
        print(f"⚠️ File not found: {EXCEL_FILE_PATH}")        
        
        
def get_possible_symptoms(predicted_diseases, csv_file_path='Dataset\dataset.csv'):
    """
    Given a list of predicted diseases and a path to a CSV file,
    this function returns a list of possible symptoms for the diseases
    found in the dataset.

    Args:
        predicted_diseases (list): List of disease names predicted by algorithms.
        csv_file_path (str): Path to the CSV file containing disease-symptom data.

    Returns:
        list: List of formatted possible symptoms based on the diseases.
    """
    # Load the dataset from CSV
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

    # Step 1: Get unique diseases
    unique_diseases = list(set(predicted_diseases))

    # Step 2: Initialize a dictionary to store symptoms for each disease
    disease_symptoms = {}

    # Step 3: Loop through each unique disease
    for disease_name in unique_diseases:
        disease_rows = df[df['label_dis'] == disease_name]

        if disease_rows.empty:
            # If no rows for this disease, skip it
            continue

        # Drop the label column to focus only on symptom data
        symptom_data = disease_rows.drop(columns=['label_dis'])

        # Get a set of all symptoms for the current disease
        disease_symptoms[disease_name] = set(symptom_data.columns[symptom_data.any(axis=0)])

    # Step 4: Collect all unique symptoms across all diseases in predicted_diseases
    possible_more_symptoms = set()

    # Step 5: Add symptoms of each disease to possible_more_symptoms
    for disease_name in unique_diseases:
        possible_more_symptoms.update(disease_symptoms[disease_name])

    # Step 6: Convert to a sorted list and optionally format with single quotes
    possible_more_symptoms = sorted(list(possible_more_symptoms))
    formatted_symptoms = [f"'{symptom}'" for symptom in possible_more_symptoms]

    return formatted_symptoms

def load_all_symptoms():
    # Load the CSV file
    csv_file_path = 'Dataset/dataset.csv'
    df = pd.read_csv(csv_file_path)

    # Get symptom columns (skip the first column if it's 'label_dis')
    symptom_columns = df.columns[1:].tolist()  # Assumes first column is label_dis

    return symptom_columns



def fetch_and_merge_array__from_excel(var1_name="user_symptoms_1", var2_name="user_symptoms_2"):
    """
    Fetches values of two variables from an Excel file by simply looking up the values
    from the 'Value' column based on the 'Variable Name' column.

    Parameters:
    - var1_name: Name of the first variable (e.g., 'user_symptoms-1')
    - var2_name: Name of the second variable (e.g., 'user_symptoms-2')

    Returns:
    - tuple: containing the values of the two variables directly from the 'Value' column
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(EXCEL_FILE_PATH , engine='openpyxl')
        
        # Fetch the values of the two variables from the 'Value' column
        var1_value = df.loc[df['Variable Name'] == var1_name, 'Value'].values
        var2_value = df.loc[df['Variable Name'] == var2_name, 'Value'].values
        
        # Check if the variables were found
        if var1_value.size == 0 or var2_value.size == 0:
            print(f"One or both of the variables '{var1_name}' or '{var2_name}' not found.")
            return None
        
        # Convert the string representation of the list into an actual list using ast.literal_eval
        var1_list = ast.literal_eval(var1_value[0])
        var2_list = ast.literal_eval(var2_value[0])
        
        # Merge the two lists
        merged_list = var1_list + var2_list
        # print(merged_list)
        
        save_array_to_excel('final_symptoms', merged_list)
        
        # Return the merged list
        return merged_list
    
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None, None



def fetch_values_from_excel(var_name="final_diseases"):
    """
    Fetches the merged values of a variable from the Excel file, 
    which was saved earlier (like 'final_symptoms').

    Parameters:
    - var_name: Name of the variable (e.g., 'final_symptoms')

    Returns:
    - list: values directly from the 'Value' column
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')
        
        # Fetch the value of the merged variable from the 'Value' column
        var_value = df.loc[df['Variable Name'] == var_name, 'Value'].values
        
        # Check if the variable was found
        if var_value.size == 0:
            print(f"Variable '{var_name}' not found.")
            return None
        
        # Convert the string representation of the list into an actual list using ast.literal_eval
        var_list = ast.literal_eval(var_value[0])
        
        # Return the values as a list
        return var_list
    
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None
    
    
def save_appointment(full_name, email, appointment_date, department, phone_number, message):
    excel_file = os.path.join("appointments.xlsx")  # Cross-platform safe path

    # Prepare the new row as a DataFrame
    new_row = pd.DataFrame([{
        "Full Name": full_name,
        "Email": email,
        "Date": appointment_date,
        "Department": department,
        "Phone": phone_number,
        "Message": message
    }])

    try:
        if os.path.exists(excel_file):
            # Read existing file and concatenate the new row
            existing_df = pd.read_excel(excel_file, engine='openpyxl')
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            # File doesn't exist, use only the new row
            updated_df = new_row

        # Save to Excel
        updated_df.to_excel(excel_file, index=False, engine='openpyxl')
        print("✅ Appointment saved to:", excel_file)
    except Exception as e:
        print("❌ Error saving appointment:", str(e))
    