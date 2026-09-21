import torch
import json

print("🧠 Membuat Synthetic Fly-Connectome Subgraph...")

# Simulasi 50 neuron (5 sensor input raycast, 40 interneuron, 5 motor output)
num_neurons = 50

# Membuat matriks keterhubungan biologis (Sparse Matrix)
torch.manual_seed(42)  # Agar repeatable
weight_matrix = torch.randn(num_neurons, num_neurons) * 0.4
weight_matrix[weight_matrix < 0.15] = 0.0  # Putus koneksi yang lemah (sinapsis jarang)
weight_matrix = torch.abs(weight_matrix)

# Simpan matriks ke berkas lokal
torch.save(weight_matrix, 'fly_subgraph_weights.pt')

# Pemetaan Indeks Neuron
metadata = {
    "sensor_inputs": [0, 1, 2, 3, 4],  # Raycast: Kiri-Jauh, Kiri, Depan, Kanan, Kanan-Jauh
    "motor_outputs": {
        "steering": 45,  # Nilai kemudi (-1 belok kiri, +1 belok kanan)
        "thrust": 46     # Kecepatan maju (0 diam, 1 maju full)
    }
}

with open('neuron_map.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Matriks SNN Sintetis ({num_neurons}x{num_neurons}) berhasil dibuat!")
print("💾 File disimpan: 'fly_subgraph_weights.pt' dan 'neuron_map.json'")
