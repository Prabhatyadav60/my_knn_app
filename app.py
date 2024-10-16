from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    age = int(request.form['age'])
    salary = int(request.form['salary'])

    # Scale the input features
    scaled_input = scaler.transform([[age, salary]])

    # Make a prediction
    prediction = model.predict(scaled_input)

    # Prepare the output message
    result = "Purchased" if prediction[0] == 1 else "Not Purchased"

    # Render the result on the webpage
    return render_template('index.html', prediction_text=f'The prediction is: {result}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
