import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from flask import Flask, render_template, request

#create a Flask instance
app = Flask(__name__)

#Read the dataset
df=pd.read_csv('dataset/CabFirmCaseStudyMerged.csv')

#To display the list of Cities,Company,Gender in the dropdown
unique_cities = df['City'].unique().tolist()
unique_companies = df['Company'].unique().tolist()
unique_genders = df['Gender'].unique().tolist()

# Encoding categorical variables
le_city = LabelEncoder()
le_company = LabelEncoder()
le_gender = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['City'])
df['Company_encoded'] = le_company.fit_transform(df['Company'])
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])

#Feature selection (dependent and independent variables)
x = df[['KM_Travelled', 'City_encoded', 'Company_encoded', 'Gender_encoded']]
y = df['Price_Charged']

#Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training the model using Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_regressor.fit(x_train, y_train)

# Save the trained model as a joblib file
joblib.dump(rf_regressor, 'models/model.joblib')

# Load model
model = joblib.load('models/model.joblib')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html', unique_cities=unique_cities, unique_companies=unique_companies, unique_genders=unique_genders,
                           error="Please enter a value for Km to Travel.")

# Define route for predict page
@app.route('/predict', methods=['POST'])
def predict():

    # Get input values from the form
    city = request.form['city']
    km_to_travel = float(request.form['km_to_travel'])
    gender = request.form['gender']
    company = request.form['company']
    
    # Encoding categorical variables
    city_encoded = le_city.transform([city])[0]
    gender_encoded = le_gender.transform([gender])[0]
    company_encoded = le_company.transform([company])[0]
    
    # Make prediction
    prediction = model.predict([[km_to_travel, city_encoded, company_encoded, gender_encoded]])
    rounded_prediction=round(prediction[0],2)

    # Render the prediction result template with the predicted price
    return render_template('index.html', prediction=rounded_prediction,unique_cities=unique_cities, unique_companies=unique_companies, unique_genders=unique_genders,
                           city=city, km_to_travel=km_to_travel, gender=gender, company=company)
prediction = None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)