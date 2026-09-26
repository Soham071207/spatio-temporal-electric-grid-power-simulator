import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpainElectricityDataset
from baselines import PersistenceModel, LSTMModel
from gnn import GATForecaster
import numpy as np
import json

class HolidayBoostLoss(nn.Module):
    """
    Asymmetric loss that penalizes under-prediction more heavily (grid safety)
    and applies a holiday boost multiplier to force the model to learn holiday patterns.
    """
    def __init__(self, under_penalty=2.0, over_penalty=1.0, holiday_boost=5.0, weekend_boost=3.0):
        super(HolidayBoostLoss, self).__init__()
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty
        self.holiday_boost = holiday_boost
        self.weekend_boost = weekend_boost
        
    def forward(self, y_pred, y_true, y_holiday, y_weekend=None):
        # diff < 0 means y_pred < y_true (under-predicting → risk of blackout)
        diff = y_pred - y_true
        squared_error = diff ** 2
        
        # Base asymmetric penalty
        weights = torch.where(diff < 0, self.under_penalty, self.over_penalty)
        base_loss = weights * squared_error
        
        # Holiday boost: y_holiday is raw binary 0/1 (NOT scaled)
        boosts = torch.where(y_holiday > 0.5, self.holiday_boost, 1.0)
        
        # Weekend boost
        if y_weekend is not None:
            boosts = torch.where((y_weekend > 0.5) & (y_holiday <= 0.5), self.weekend_boost, boosts)
        
        return torch.mean(base_loss * boosts)

def calculate_metrics(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = np.sqrt(mse)
    return mse, rmse, mae

def evaluate(model, dataloader, criterion, edge_index=None, edge_weight=None, is_gnn=False, is_persistence=False, device='cpu'):
    if not is_persistence:
        model.eval()
        
    total_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x, y, y_holiday, y_weekend in dataloader:
            if not is_persistence:
                x = x.to(device)
                y = y.to(device)
                y_holiday = y_holiday.to(device)
                y_weekend = y_weekend.to(device)

            if is_persistence:
                pred = model.predict(x)
            elif is_gnn:
                pred = model(x, edge_index.to(device), edge_weight.to(device))
            else:
                pred = model(x)
                
            if is_persistence:
                loss = nn.functional.mse_loss(pred, y)  # Persistence uses standard MSE
            else:
                loss = criterion(pred, y, y_holiday, y_weekend)
            total_loss += loss.item() * x.size(0)
            
            all_y_true.append(y)
            all_y_pred.append(pred)
            
    avg_loss = total_loss / len(dataloader.dataset)
    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)
    
    mse, rmse, mae = calculate_metrics(y_true, y_pred)
    return avg_loss, rmse, mae

def train_epoch(model, dataloader, optimizer, criterion, edge_index=None, edge_weight=None, is_gnn=False, device='cpu'):
    model.train()
    total_loss = 0
    
    for x, y, y_holiday, y_weekend in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_holiday = y_holiday.to(device)
        y_weekend = y_weekend.to(device)
        
        optimizer.zero_grad()
        if is_gnn:
            pred = model(x, edge_index.to(device), edge_weight.to(device))
        else:
            pred = model(x)
            
        loss = criterion(pred, y, y_holiday, y_weekend)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(dataloader.dataset)

def run_experiment():
    print("Loading datasets (2015-2026)...")
    try:
        train_dataset = SpainElectricityDataset(split="train", year_start=2015)
        val_dataset = SpainElectricityDataset(split="val", year_start=2015)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return
        
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Model parameters
    in_channels = 16  # demand(1) + weather(4) + pop_density(1) + industry(1) + price(1) + time_encodings(6) + holiday(1) + weekend(1)
    hidden_channels = 64
    pred_horizon = 24
    
    edge_index = train_dataset.edge_index
    edge_weight = train_dataset.edge_weight
    
    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Persistence Baseline
    print("\n--- Evaluating Persistence Baseline ---")
    persistence = PersistenceModel(pred_horizon)
    _, rmse, mae = evaluate(persistence, val_loader, None, is_persistence=True)
    print(f"Persistence -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # 2. GAT Forecaster with Holiday-Boosted Asymmetric Loss
    print("\n--- Training GAT Forecaster with HolidayBoost Loss ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    gnn_model = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4, dropout=0.2).to(device)
    
    best_model_path = os.path.join(models_dir, "best_gat_forecaster_asymmetric.pth")
    start_epoch = 1
    best_val_loss = float('inf')
    
    if os.path.exists(best_model_path):
        print(f"Resuming from checkpoint: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            gnn_model.load_state_dict(checkpoint['state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Loaded weights from Epoch {start_epoch-1} with Val Loss: {best_val_loss:.4f}")
        else:
            gnn_model.load_state_dict(checkpoint)
            
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.001, weight_decay=1e-5) 
    
    # Cosine Annealing with Warm Restarts for smoother LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_gnn, T_0=20, T_mult=2, eta_min=1e-6)
    
    patience_counter = 0
    patience_limit = 60  # Increased patience to ensure it runs to 100+ epochs
    
    # Holiday-Boosted Asymmetric Loss (under_penalty=2.0 for grid safety)
    criterion = HolidayBoostLoss(under_penalty=2.0, over_penalty=1.0, holiday_boost=5.0, weekend_boost=3.0)
    
    num_epochs = 100  # Cap at 100 as per user request
    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_epoch(gnn_model, train_loader, optimizer_gnn, criterion, edge_index, edge_weight, is_gnn=True, device=device)
        val_loss, rmse, mae = evaluate(gnn_model, val_loader, criterion, edge_index, edge_weight, is_gnn=True, device=device)
        
        scheduler.step()
        current_lr = optimizer_gnn.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save full checkpoint with metadata
            checkpoint = {
                'state_dict': gnn_model.state_dict(),
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'pred_horizon': pred_horizon,
                'heads': 4,
                'dropout': 0.2,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'loss_config': {
                    'under_penalty': 2.0,
                    'over_penalty': 1.0,
                    'holiday_boost': 5.0,
                },
                'num_scaled_features': 7,  # Features 0-6 are scaled (demand, temp, humidity, wind, solar, pop_density, industry)
            }
            torch.save(checkpoint, best_model_path)
            print(f"  -> Saved new best model (epoch {epoch})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    run_experiment()
