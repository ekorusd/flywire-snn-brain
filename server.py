from flask import Flask, request, jsonify
from flask_cors import CORS
from fly_brain_engine import FlyBrainEngine

app = Flask(__name__)
CORS(app)

# Inisialisasi engine SNN otak lalat
brain = FlyBrainEngine()

@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    if not data or 'raycasts' not in data:
        return jsonify({'error': 'Missing raycasts input'}), 400
        
    raycasts = data['raycasts']
    action = brain.process_sensor_inputs(raycasts)
    
    return jsonify({
        'status': 'success',
        'action': action
    })

if __name__ == '__main__':
    print('🚀 Fly-Brain AI Server berjalan di http://localhost:5000')
    app.run(host='0.0.0.0', port=5000)