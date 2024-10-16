# 🌟 K-Nearest Neighbors (KNN) Model for Purchase Prediction 

This project implements a K-Nearest Neighbors (KNN) model to predict whether a customer is likely to make a purchase based on their age and salary. The model is trained using a dataset and can be easily deployed using Flask for web interface interactions.

## 📚 Table of Contents
- [✨ Features](#features)
- [🔧 Requirements](#requirements)
- [📥 Installation](#installation)
- [💻 Usage](#usage)

## ✨ Features
- Predicts if a purchase is likely based on user input (age and salary). 💰
- Scalable and can be integrated into larger applications. 📈
- Web interface built using Flask. 🌐

## 🔧 Requirements
To run this project, you need to have the following installed:
- Python 3.6 or higher 🐍
- Flask 🌸
- Scikit-learn 📊
- NumPy ➕
- Pandas 🐼
- Pickle 📦

You can install the required libraries using pip:
```bash
pip install flask scikit-learn numpy pandas
📥 Installation
Clone the repository:

git clone https://github.com/your-username/knn-purchase-prediction.git
cd knn-purchase-prediction
Load the trained model and scaler: Make sure you have knn_model.pkl and scaler.pkl files in your project directory. 📂

Run the application:

python app.py
The application will run on http://127.0.0.1:5000/. 🚀
```
💻 Usage
Open your web browser and navigate to http://127.0.0.1:5000/. 🌍
Enter the customer's age and salary in the form provided. 👤💵
Click the "Predict" button to see if the customer is likely to make a purchase. 🛒
