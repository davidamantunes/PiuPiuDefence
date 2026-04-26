# PiuPiuDefense

## Overview
**PiuPiuDefense** is an advanced AI-driven system designed to mitigate multiple simultaneous and diverse Unmanned Aerial System (UAS) threats. 

### The Challenge
In modern defense scenarios, opponents frequently mix low-cost and high-end UAS threats. This tactic is deliberately designed to trick defenders into misusing expensive countermeasures on insignificant targets, ultimately exhausting and wasting critical defensive capacity over time.

### Our Approach
We designed a methodology tackling this resource-exhaustion issue using a blend of Machine Learning and rule-based decision mechanisms:
1. **AI Threat Prediction:** We use sensor and telemetry data to train an AI model (MLP) to predict a threat's danger level and potential blast/damage radius.
2. **Decision & Prioritization:** Based on the AI predictions, the system prioritizes the threats and decides on the most cost-effective countermeasure.
3. **Simulation:** A full simulation generated to visualize the incoming threats, target trajectories, and the deployed countermeasures.

### Business Model
* **Model:** B2B subscription-based delivery tailored for defense operations.
* **Customers:** Military entities and Government defense sectors.
* **Value Proposition:** Enabling defense forces to significantly reduce operational costs by saving high-tier countermeasures for truly critical incoming threats.

## Architecture & Project Structure

The project is structured into modular components mapping directly to the threat mitigation pipeline.

* **`Engine/`** - Core coordination mechanisms integrating the prediction models and simulation logic.
* **`GUI/`** - A Streamlit-based web application serving as the primary system interface (`app.py`).
* **`NN_Model/`** - Machine Learning components:
  * PyTorch-based Multilayer Perceptron (`ThreatMLP.py`, `MLP_Training.py`).
  * Functionality to convert the PyTorch model to ONNX for fast inference (`Torch_to_ONNX.py`).
* **`RuleBased_Decision/`** - The deterministic logic block (`RB_Decision.py`) that decides on the exact counter-deployment mechanism based on the Neural Network's categorization.
* **`SimLogic/`** - The backend physics and interaction simulation written in C++ for performance (`main.cpp`, `sim.hpp`).
* **`SimGui/`** - Python tools (`animate.py`) for visually animating the outputs of our C++ simulation.

---

## Getting Started

### Python Virtual Environment Setup

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

To leave the environment later:

```bash
deactivate
```
