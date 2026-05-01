
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = joblib.load('position_predictor.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
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
    app.run(host='0.0.0.0', port=5000)
