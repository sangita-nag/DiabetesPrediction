import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('LogisticRegression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
            
    Pregnancies = int(request.form['Pregnancies'])
    Glucose = int(request.form['Glucose'])
    BloodPressure = int(request.form['BloodPressure'])
    SkinThickness = int(request.form['SkinThickness'])
    Insulin = int(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = int(request.form['Age'])

       
    featureArray = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(featureArray)
    
    predString = ""
    
    if prediction[0] == 0:
        predString = "Non-Diabetic"
    else:
        predString = "Diabetic"
    

    return render_template('index.html', prediction_text='Diabetic or not:  {}'.format(predString))


if __name__ == "__main__":
    app.run(debug=True)
