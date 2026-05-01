
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Print current directory and files (for debugging)
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Load model
try:
    model = joblib.load('position_predictor.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data['data'], columns=[
            'accurateCrosses', 'accurateLongBalls', 'aerialDuelsWon',
            'clearances', 'interceptions', 'tacklesWon', 'goals',
            'assists', 'keyPasses', 'saves', 'shotsOnTarget', 'minutesPlayed'
        ])
        predictions = model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
