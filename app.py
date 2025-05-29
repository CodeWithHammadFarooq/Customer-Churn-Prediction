from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data_scaled = scaler.transform([data])
    prediction = model.predict(data_scaled)
    return render_template('index.html', prediction_text='Churn Prediction: {}'.format('Yes' if prediction[0] else 'No'))

if __name__ == "__main__":
    app.run(debug=True)