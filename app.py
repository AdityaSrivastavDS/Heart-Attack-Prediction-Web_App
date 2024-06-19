from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
with open('Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['age'], data['sex'], data['cp'], data['trestbps'],
                         data['chol'], data['fbs'], data['restecg'], data['thalach'],
                         data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']])
    
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
