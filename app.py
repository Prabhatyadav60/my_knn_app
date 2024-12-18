from flask import Flask, request, render_template
import pickle
import numpy as np


app = Flask(__name__)


model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    salary = int(request.form['salary'])


    scaled_input = scaler.transform([[age, salary]])

    prediction = model.predict(scaled_input)


    result = "Purchased" if prediction[0] == 1 else "Not Purchased"


    return render_template('index.html', prediction_text=f'The prediction is: {result}')


if __name__ == "__main__":
    app.run(debug=True)
