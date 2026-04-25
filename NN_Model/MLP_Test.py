import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse

from ThreatMLP import ThreatMLP_Basic, ThreatMLP_Trajectory
from MLP_Training import UAVDataset

def test_mlp(csv_path, model_path, batch_size=32, save_predictions=False):
    print(f"Loading test data from {csv_path}...")
    dataset = UAVDataset(csv_path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Infer the number of distinct UAV types from the max id in the dataset
    num_uav_types = int(dataset.df["UAV_ID"].max()) + 1

    print("Target Model: ThreatMLP_Trajectory")
    is_trajectory = True
    model = ThreatMLP_Trajectory(num_uav_types=max(num_uav_types, 5))
        
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion_threat = nn.NLLLoss()
    criterion_damage = nn.MSELoss()
    
    print("Evaluating Model on Test Set...")
    test_loss = 0.0
    correct_threats = 0
    total_samples = 0
    
    test_results = []
    
    with torch.no_grad():
        for batch in test_loader:
            if is_trajectory:
                x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                threat_probs, damage = model(x, y, speed, dx, dy, uav_type)
            else:
                x, y, speed, uav_type, target_threat, target_damage = batch
                threat_probs, damage = model(x, y, speed, uav_type)
                
            log_threat_probs = torch.log(threat_probs + 1e-8)
            t_loss = criterion_threat(log_threat_probs, target_threat) + 0.5 * criterion_damage(damage, target_damage)
            test_loss += t_loss.item() * len(target_threat)
            
            # Calculate accuracy for threat classification
            predicted_threats = torch.argmax(threat_probs, dim=1)
            correct_threats += (predicted_threats == target_threat).sum().item()
            total_samples += len(target_threat)
            
            # Record individual predictions conditionally
            if save_predictions:
                for i in range(len(target_threat)):
                    test_results.append({
                        'Predicted_Threat': predicted_threats[i].item() + 1,
                        'Predicted_Damage': damage[i].item()
                    })
            
    test_loss /= total_samples
    if total_samples > 0:
        test_accuracy = (correct_threats / total_samples) * 100.0
    else:
        test_accuracy = 0.0
        
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Threat Classification Accuracy: {test_accuracy:.2f}%")
    
    if save_predictions:
        results_path = "Outputs/test_results_predictions.csv"
        pd.DataFrame(test_results).to_csv(results_path, index=False)
        print(f"Test data and predictions saved to '{results_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test UAV Threat MLP')
    parser.add_argument('--csv_path', type=str, default='Dataset/SimOut_Test.csv', help='Path to the test dataset CSV file')
    parser.add_argument('--model_path', type=str, default='Outputs/trained_mlp_best.pth', help='Path to the trained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--save_predictions', action='store_true', help='Flag to save hold-out test predictions to CSV')
    
    args = parser.parse_args()
    
    test_mlp(
        csv_path=args.csv_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions
    )
