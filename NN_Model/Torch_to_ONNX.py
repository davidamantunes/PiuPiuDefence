import torch
from ThreatMLP import ThreatMLP_Trajectory
import os

def export_to_onnx():
    num_uav_types = 10 # Adjusted to match the checkpoint (10 uav types)
    
    # Initialize the model 
    model = ThreatMLP_Trajectory(num_uav_types=num_uav_types)
    
    # Load trained weights if available
    weights_path = "trained_mlp_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    
    model.eval()

    # Dummy inputs for ThreatMLP_Trajectory (batch_size=1)
    # x, y, speed, dx, dy shape: [1, 1], uav_type shape: [1]
    x = torch.randn(1, 1, dtype=torch.float32)
    y = torch.randn(1, 1, dtype=torch.float32)
    speed = torch.randn(1, 1, dtype=torch.float32)
    dx = torch.randn(1, 1, dtype=torch.float32)
    dy = torch.randn(1, 1, dtype=torch.float32)
    uav_type = torch.zeros(1, dtype=torch.long)

    inputs = (x, y, speed, dx, dy, uav_type)

    output_onnx_file = "Outputs/threat_mlp.onnx"
    torch.onnx.export(
        model,                  
        inputs,        
        output_onnx_file,        
        input_names=["x", "y", "speed", "dx", "dy", "uav_type"],  
        output_names=["threat_probs", "damage"],
        dynamo=True,             
        external_data=False
    )
    print(f"Model exported to {output_onnx_file}")

if __name__ == "__main__":
    export_to_onnx()
