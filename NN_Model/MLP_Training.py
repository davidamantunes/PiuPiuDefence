import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import copy
import argparse

from ThreatMLP import ThreatMLP_Basic, ThreatMLP_Trajectory

class UAVDataset(Dataset):
    def __init__(self, csv_file):
        # Read CSV. It's assumed to have a dummy header string on the first line.
        df = pd.read_csv(csv_file, header=0)
        self.df = df
        self.has_trajectory = "Tx" in df.columns and "Ty" in df.columns
        self.num_cols = 8 if self.has_trajectory else 6
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        uav_id = int(row["UAV_ID"])
        px = np.float32(row["MeasuredX"])
        py = np.float32(row["MeasuredY"])
        speed = np.float32(row["Speed"])
        # PyTorch expects 0-indexed class labels for CrossEntropy/NLLLoss
        threat_id = int(row["RiskLevel"])
        damage = np.float32(row["Damage_Potential"])
        
        if not self.has_trajectory:
            return (
                torch.tensor([px]), torch.tensor([py]), torch.tensor([speed]),
                torch.tensor(uav_id, dtype=torch.long),
                torch.tensor(threat_id, dtype=torch.long),
                torch.tensor([damage])
            )
        else:
            tx = np.float32(row["Tx"])
            ty = np.float32(row["Ty"])
            return (
                torch.tensor([px]), torch.tensor([py]), torch.tensor([speed]),
                torch.tensor([tx]), torch.tensor([ty]),
                torch.tensor(uav_id, dtype=torch.long),
                torch.tensor(threat_id, dtype=torch.long),
                torch.tensor([damage])
            )

def train_mlp(csv_path, num_epochs=50, batch_size=32, lr=0.001, test_split=0.2, k_folds=5, save_predictions=False):
    print(f"Loading data from {csv_path}...")
    dataset = UAVDataset(csv_path)
    
    # We infer the number of distinct UAV types from the max id in the dataset
    num_uav_types = int(dataset.df["UAV_ID"].max()) + 1
    
    # Split into general (train+val) and final test set
    dataset_indices = np.arange(len(dataset))
    cv_idx, test_idx = train_test_split(dataset_indices, test_size=test_split, random_state=42)
    
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Total samples: {len(dataset)} | CV (Train+Val) samples: {len(cv_idx)} | Test samples: {len(test_idx)}")
    
    if dataset.num_cols == 6:
        print("Detected 6 columns. Target Model: ThreatMLP_Basic")
        is_trajectory = False
    else:
        print("Detected 8 columns. Target Model: ThreatMLP_Trajectory")
        is_trajectory = True
        
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    criterion_threat = nn.NLLLoss()
    criterion_damage = nn.MSELoss()
    
    best_overall_model_weights = None
    best_overall_val_loss = float('inf')
    
    print("Starting K-Fold Cross Validation...")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(cv_idx)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        
        # Create dataloaders for the current fold
        fold_train_idx = cv_idx[train_ids]
        fold_val_idx = cv_idx[val_ids]
        
        train_sub = Subset(dataset, fold_train_idx)
        val_sub = Subset(dataset, fold_val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        
        # Initialize a fresh model for each fold
        if not is_trajectory:
            model = ThreatMLP_Basic(num_uav_types=max(num_uav_types, 10))
        else:
            model = ThreatMLP_Trajectory(num_uav_types=max(num_uav_types, 10))
            
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_fold_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            # Training Loop
            for batch in train_loader:
                optimizer.zero_grad()
                
                if is_trajectory:
                    x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                    threat_probs, damage = model(x, y, speed, dx, dy, uav_type)
                else:
                    x, y, speed, uav_type, target_threat, target_damage = batch
                    threat_probs, damage = model(x, y, speed, uav_type)
                    
                log_threat_probs = torch.log(threat_probs + 1e-8)
                loss = criterion_threat(log_threat_probs, target_threat) + 0.5 * criterion_damage(damage, target_damage)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(target_threat)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if is_trajectory:
                        x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                        threat_probs, damage = model(x, y, speed, dx, dy, uav_type)
                    else:
                        x, y, speed, uav_type, target_threat, target_damage = batch
                        threat_probs, damage = model(x, y, speed, uav_type)
                        
                    log_threat_probs = torch.log(threat_probs + 1e-8)
                    v_loss = criterion_threat(log_threat_probs, target_threat) + 0.5 * criterion_damage(damage, target_damage)
                    val_loss += v_loss.item() * len(target_threat)
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                if val_loss < best_overall_val_loss:
                    best_overall_val_loss = val_loss
                    best_overall_model_weights = copy.deepcopy(model.state_dict())
            
            # Print occasionally
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
        print(f"Fold {fold + 1} best Val Loss: {best_fold_val_loss:.4f}")
        
    print("\n--- Cross Validation Finished ---")
    
    # Load the best overall architecture weights to test
    if not is_trajectory:
        best_model = ThreatMLP_Basic(num_uav_types=max(num_uav_types, 10))
    else:
        best_model = ThreatMLP_Trajectory(num_uav_types=max(num_uav_types, 10))
    
    best_model.load_state_dict(best_overall_model_weights)
    best_model.eval()
    
    print("Evaluating Best Model on Hold-Out Test Set...")
    test_loss = 0.0
    correct_threats = 0
    total_samples = 0
    
    test_results = []
    
    with torch.no_grad():
        for batch in test_loader:
            if is_trajectory:
                x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                threat_probs, damage = best_model(x, y, speed, dx, dy, uav_type)
            else:
                x, y, speed, uav_type, target_threat, target_damage = batch
                threat_probs, damage = best_model(x, y, speed, uav_type)
                
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
                    if is_trajectory:
                        test_results.append({
                            'UAV_id': uav_type[i].item(),
                            'Px': x[i].item(),
                            'Py': y[i].item(),
                            'Tx': dx[i].item(),
                            'Ty': dy[i].item(),
                            'Speed': speed[i].item(),
                            'Actual_Threat': target_threat[i].item() + 1,  # Undo 0-index offset
                            'Predicted_Threat': predicted_threats[i].item() + 1,
                            'Actual_Damage': target_damage[i].item(),
                            'Predicted_Damage': damage[i].item()
                        })
                    else:
                        test_results.append({
                            'UAV_id': uav_type[i].item(),
                            'Px': x[i].item(),
                            'Py': y[i].item(),
                            'Speed': speed[i].item(),
                            'Actual_Threat': target_threat[i].item() + 1,  # Undo 0-index offset
                            'Predicted_Threat': predicted_threats[i].item() + 1,
                            'Actual_Damage': target_damage[i].item(),
                            'Predicted_Damage': damage[i].item()
                        })
            
    test_loss /= total_samples
    test_accuracy = (correct_threats / total_samples) * 100.0
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Threat Classification Accuracy: {test_accuracy:.2f}%")
    
    if save_predictions:
        results_path = "test_results_predictions.csv"
        pd.DataFrame(test_results).to_csv(results_path, index=False)
        print(f"Test data and predictions saved to '{results_path}'")
    
    save_path = "trained_mlp_best.pth"
    torch.save(best_overall_model_weights, save_path)
    print(f"Best model saved to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UAV Threat MLP')
    parser.add_argument('--csv_path', type=str, default='Dataset/SimOut.csv', help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--save_predictions', action='store_true', help='Flag to save hold-out test predictions to CSV')
    
    args = parser.parse_args()
    
    train_mlp(
        csv_path=args.csv_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions
    )
