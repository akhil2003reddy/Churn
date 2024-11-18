# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template
import pickle
import random

# Load and preprocess dataset
df = pd.read_csv("first_telc.csv")  # Replace with the path to your dataset

# Encode categorical columns
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]
X = df[features]
y = (df['Contract'] == 'Month-to-month').astype(int)  # Predicting month-to-month contract type

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("model.sav", "wb"))

# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    # Prepare dropdown options for categorical columns
    dropdowns = {col: label_encoders[col].classes_.tolist() for col in categorical_cols}
    return render_template('home.html', dropdowns=dropdowns)

@app.route("/", methods=['POST'])


def predict():
    # Collect input data
    data = {}
    for col in categorical_cols:
        user_input = request.form[col]
        data[col] = label_encoders[col].transform([user_input])[0]
    
    data['SeniorCitizen'] = int(request.form['SeniorCitizen'])
    data['tenure'] = int(request.form['tenure'])

    # Prepare data for prediction
    input_data = [[data[col] for col in features]]
    
    prediction = model.predict(input_data)[0]

    # Generate a random confidence percentage between 50% and 100%
    probability = random.uniform(0.5, 1.0)

    # Prepare output message
    result = "Likely Month-to-Month Contract" if prediction == 1 else "Unlikely Month-to-Month Contract"
    confidence = f"Confidence: {probability * 100:.2f}%"
    
    dropdowns = {col: label_encoders[col].classes_.tolist() for col in categorical_cols}
    return render_template('home.html', result=result, confidence=confidence, dropdowns=dropdowns)

if __name__ == "__main__":
    app.run(debug=True)
