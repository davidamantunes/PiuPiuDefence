import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import copy
import argparse

from ThreatMLP import ThreatMLP_Trajectory

class UAVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=0)
        self.df = df
        self.has_trajectory = "Tx" in df.columns and "Ty" in df.columns
        self.num_cols = 8 if self.has_trajectory else 6
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        uav_id = int(row["UAV_ID"])
        px = np.float32(row["MeasuredX_Normalized"])
        py = np.float32(row["MeasuredY_Normalized"])
        speed = np.float32(row["Speed_Normalized"])
        threat_id = int(row["RiskLevel"])
        damage = np.float32(row["Damage_Potential"])
        tx = np.float32(row["Tx"])
        ty = np.float32(row["Ty"])
        return (
            torch.tensor([px]), torch.tensor([py]), torch.tensor([speed]),
            torch.tensor([tx]), torch.tensor([ty]),
            torch.tensor(uav_id, dtype=torch.long),
            torch.tensor(threat_id, dtype=torch.long),
            torch.tensor([damage])
        )

def train_mlp(csv_path, num_epochs=100, batch_size=32, lr=0.001, k_folds=5):
    print(f"Loading data from {csv_path}...")
    dataset = UAVDataset(csv_path)
    
    # We infer the number of distinct UAV types from the max id in the dataset
    num_uav_types = int(dataset.df["UAV_ID"].max()) + 1
    
    # Use full dataset for CV
    dataset_indices = np.arange(len(dataset))
    cv_idx = dataset_indices
    
    print(f"Total samples: {len(dataset)} | CV (Train+Val) samples: {len(cv_idx)}")
        
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
        model = ThreatMLP_Trajectory(num_uav_types=max(num_uav_types, 5))
            
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_fold_val_loss = float('inf')
        
        patience = 5  # Stop if no improvement after 5 epochs
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            # Training Loop
            for batch in train_loader:
                optimizer.zero_grad()
                
                x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                threat_probs, damage = model(x, y, speed, dx, dy, uav_type)
                    
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
                    x, y, speed, dx, dy, uav_type, target_threat, target_damage = batch
                    threat_probs, damage = model(x, y, speed, dx, dy, uav_type)
                        
                    log_threat_probs = torch.log(threat_probs + 1e-8)
                    v_loss = criterion_threat(log_threat_probs, target_threat) + 0.5 * criterion_damage(damage, target_damage)
                    val_loss += v_loss.item() * len(target_threat)
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                epochs_no_improve = 0
                if val_loss < best_overall_val_loss:
                    best_overall_val_loss = val_loss
                    best_overall_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}!")
                break
            # Print occasionally
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
        print(f"Fold {fold + 1} best Val Loss: {best_fold_val_loss:.4f}")
        
    print("\n--- Cross Validation Finished ---")
    
    save_path = "Outputs/trained_mlp_best.pth"
    torch.save(best_overall_model_weights, save_path)
    print(f"Best model saved to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UAV Threat MLP')
    parser.add_argument('--csv_path', type=str, default='Dataset/SimOut.csv', help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')
    
    args = parser.parse_args()
    
    train_mlp(
        csv_path=args.csv_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
