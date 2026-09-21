import os
import pandas as pd
import torch
from caveclient import CAVEclient

# 1. Inisialisasi CAVE Client dengan server_address resmi FlyWire
datastack_name = "flywire_fafb_production"
server_url = "https://global.v1.cave-connectome.org"

client = CAVEclient(datastack_name, server_address=server_url)

# Set token yang sudah didapat
token = "91025d225da8326e6535398982ff860d"
client.auth.token = token

print("🔌 Terhubung ke FlyWire API...")

# 2. Ambil sampel data sinapsis
print("📥 Mengunduh sampel data konektivitas neuron...")
synapses_df = client.materialize.query_table('synapses_nt_v1', limit=1000)

print(f"✅ Berhasil mengunduh {len(synapses_df)} koneksi sinapsis.")
print("\nBeberapa baris pertama data sinapsis:")
print(synapses_df[['pre_pt_root_id', 'post_pt_root_id', 'syn_weight']].head())

# 3. Konversi ke Matriks Bobot PyTorch (Adjacency Matrix)
unique_pre = synapses_df['pre_pt_root_id'].unique()
unique_post = synapses_df['post_pt_root_id'].unique()
all_neurons = list(set(unique_pre).union(set(unique_post)))

neuron_to_idx = {n_id: idx for idx, n_id in enumerate(all_neurons)}
num_neurons = len(all_neurons)

weight_matrix = torch.zeros((num_neurons, num_neurons))

for _, row in synapses_df.iterrows():
    src = neuron_to_idx[row['pre_pt_root_id']]
    dst = neuron_to_idx[row['post_pt_root_id']]
    weight = row.get('syn_weight', 1.0)
    weight_matrix[src, dst] += weight

print(f"\n🧠 Matriks Konektivitas SNN berhasil dibuat!")
print(f"Ukuran Matriks: {weight_matrix.shape} (Jumlah Neuron: {num_neurons})")

torch.save(weight_matrix, 'fly_subgraph_weights.pt')
print("💾 Matriks disimpan ke 'fly_subgraph_weights.pt'")
