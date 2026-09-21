import torch
import json

class FlyBrainEngine:
    def __init__(self, weights_path='fly_subgraph_weights.pt', map_path='neuron_map.json'):
        # Load matriks bobot sinapsis
        self.weights = torch.load(weights_path)
        with open(map_path, 'r') as f:
            self.map = json.load(f)
            
        self.num_neurons = self.weights.shape[0]
        # State potensial membran neuron (voltase awal 0)
        self.potentials = torch.zeros(self.num_neurons)
        self.decay = 0.8  # Leaky Integrate-and-Fire decay rate

    def process_sensor_inputs(self, raycast_distances):
        """
        raycast_distances: List 5 angka (0.0 = dekat/menabrak, 1.0 = jauh/aman)
        Contoh: [1.0, 1.0, 0.1, 1.0, 1.0] -> Ada rintangan dekat di DEPAN!
        """
        # Invert jarak: semakin dekat rintangan, semakin tinggi sinyal pemicu saraf
        sensor_signals = 1.0 - torch.tensor(raycast_distances, dtype=torch.float32)
        
        # 1. Masukkan sinyal sensor ke input layer
        input_indices = self.map['sensor_inputs']
        self.potentials[input_indices] += sensor_signals

        # 2. Merambatkan sinyal melalui matriks bobot sinapsis
        self.potentials = torch.matmul(self.weights, self.potentials) * self.decay

        # 3. Baca output dari neuron motorik
        steer_idx = self.map['motor_outputs']['steering']
        thrust_idx = self.map['motor_outputs']['thrust']

        # Normalisasi output aksi
        steering = torch.tanh(self.potentials[steer_idx]).item()  # [-1.0 belok kiri, +1.0 belok kanan]
        thrust = torch.sigmoid(self.potentials[thrust_idx]).item() # [0.0 stop, 1.0 maju full]

        return {"steering": round(steering, 3), "thrust": round(thrust, 3)}

# TES SIMULASI LOKAL
if __name__ == "__main__":
    brain = FlyBrainEngine()
    
    # Tes Skenario: Rintangan dekat di sebelah kanan [Kiri_Jauh, Kiri, Depan, Kanan, Kanan_Jauh]
    test_sensor = [1.0, 1.0, 0.8, 0.1, 0.2] 
    
    action = brain.process_sensor_inputs(test_sensor)
    print("📡 Input Raycast Sensor (Rintangan di kanan):", test_sensor)
    print("🧠 Output Refleks Otak Lalat:", action)