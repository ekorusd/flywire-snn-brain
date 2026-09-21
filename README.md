# 🧠 FlyWire SNN Brain Engine for Autonomous Game NPCs

A Lightweight Spiking Neural Network (SNN) Proof of Concept (PoC) derived from the *Drosophila melanogaster* (fruit fly) brain connectome data provided by **FlyWire/CAVE**. 

This repository translates biological synaptic connectivity into an actionable neural control engine, driving autonomous, organic obstacle avoidance behaviors for game NPCs and flying drones in real-time.

---

## 🚀 Key Features

- **Biological Connectome Integration:** Extracts synaptic connectivity matrices (Adjacency Graphs) from FlyWire dataset via `caveclient`.
- **Spiking Neural Network (SNN) Dynamics:** Uses Leaky Integrate-and-Fire (LIF) neural models to calculate real-time motor signals from spatial raycast inputs without manual `if-else` rules.
- **Cross-Engine Compatibility:** Includes a lightweight Flask REST API (`server.py`) serving HTTP endpoints for **Roblox Studio (Luau)**, **Unity (C#)**, and **Unreal Engine (C++/Blueprints)**.
- **Low Latency & Lightweight:** Optimized using `PyTorch` matrix operations to maintain sub-millisecond inference times.

---

## 🛠️ Architecture Overview

```text
[ Game Engine (Roblox/Unity) ]
      │  (Raycast Sensor Array)
      ▼  POST /step {"raycasts": [1.0, 1.0, 0.2, 1.0, 1.0]}
[ Flask API Bridge ]
      │
      ▼
[ SNN Fly Brain Engine ]
      │  - Synaptic Propagation (torch.matmul)
      │  - LIF Decay Dynamics
      ▼
[ Output Action Signal ]  ==> {"steering": 0.377, "thrust": 0.581}

💻 Quick Start
1. Clone & Setup Environment
git clone [https://github.com/ekorusd/flywire-snn-brain.git](https://github.com/ekorusd/flywire-snn-brain.git)
cd flywire-snn-brain

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

2. Generate Connectome Graph & Test SNN Engine
# Generate localized SNN connectivity weights
python fetch_dummy_connectome.py

# Test local neural propagation
python fly_brain_engine.py

3. Run the Inference API Server
python server.py

🎮 Game Engine Integration
Roblox Studio (Luau)
Enable Allow HTTP Requests under Game Settings -> Security.

Attach a script to your NPC Model sending 5-directional Raycasts to http://127.0.0.1:5000/step.

Apply returned steering and thrust values to the NPC's CFrame.

📜 License & Attribution
Connectome Data Source: FlyWire / Codex (Princeton University / Seung Lab).

License: Open Access under Creative Commons Attribution (CC BY 4.0).
