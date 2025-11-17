from django.shortcuts import render
from django.http import JsonResponse
import joblib
import numpy as np
import os
import json

# Load the pre-trained model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'svm_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    model = None
    scaler = None
    print("Error: Model or scaler files not found. Please run the training script.")

def predict(request):
    if request.method == 'POST':
        # This handles the asynchronous fetch request from the JavaScript
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)
            features = [
                data['anxiety_level'],
                data['self_esteem'],
                data['mental_health_history'],
                data['depression'],
                data['headache'],
                data['blood_pressure'],
                data['sleep_quality'],
                data['breathing_problem'],
                data['noise_level'],
                data['living_conditions'],
                data['safety'],
                data['basic_needs'],
                data['academic_performance'],
                data['study_load'],
                data['teacher_student_relationship'],
                data['future_career_concerns'],
                data['social_support'],
                data['peer_pressure'],
                data['extracurricular_activities'],
                data['bullying'],
            ]

            # Convert to numpy array, scale, and predict
            input_data_as_numpy_array = np.asarray(features).reshape(1, -1)
            scaled_data = scaler.transform(input_data_as_numpy_array)
            prediction_value = model.predict(scaled_data)[0]

            # Map the numeric prediction to a human-readable label
            labels = {0: 'Normal', 1: 'Stressed', 2: 'At Risk'}
            prediction = labels.get(prediction_value, 'Unknown')

            # Return the prediction as a JSON response
            return JsonResponse({'prediction': prediction})
        except (KeyError, json.JSONDecodeError):
            return JsonResponse({'error': 'Invalid data format'}, status=400)

    # This handles the initial GET request for the page
    return render(request, 'predictor/predict.html')