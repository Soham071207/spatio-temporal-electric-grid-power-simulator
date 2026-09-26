import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATForecaster(nn.Module):
    """
    A Spatio-Temporal Graph Attention Network.
    Uses GAT for spatial relationships and LSTM for temporal sequences.
    Includes BatchNorm for training stability and Dropout for regularization.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, pred_horizon, heads=4, dropout=0.2):
        super(GATForecaster, self).__init__()
        self.pred_horizon = pred_horizon
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.out_channels = out_channels
        
        # Spatial Encoder: 2 layers of GAT with BatchNorm
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Temporal Encoder
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=2, batch_first=True, dropout=dropout)
        
        # Predictor (Output size out_channels * pred_horizon)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, pred_horizon * out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (Batch, Nodes, SeqLen, Features)
        edge_index: (2, NumEdges) from adjacency matrix
        """
        B, N, S, F_in = x.shape
        
        # Transpose to (Batch, SeqLen, Nodes, Features)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # We need to process B * S identical graphs.
        num_graphs = B * S
        
        # Flatten x to 2D: (B * S * N, F_in)
        x_flat = x.view(num_graphs * N, F_in)
        
        # Batch the edge_index
        E = edge_index.size(1)
        edge_index_batched = edge_index.repeat(1, num_graphs) # (2, NumEdges * num_graphs)
        offsets = torch.arange(0, num_graphs * N, N, device=edge_index.device).view(1, -1)
        offsets = offsets.repeat_interleave(E, dim=1)
        edge_index_batched = edge_index_batched + offsets
        
        # Apply GAT layers with BatchNorm
        gcn_out = self.gat1(x_flat, edge_index_batched)
        gcn_out = self.bn1(gcn_out)
        gcn_out = F.elu(gcn_out)
        gcn_out = F.dropout(gcn_out, p=self.dropout, training=self.training)
        
        gcn_out = self.gat2(gcn_out, edge_index_batched)
        gcn_out = self.bn2(gcn_out)
        gcn_out = F.elu(gcn_out)
        
        # Reshape back to (Batch, SeqLen, Nodes, Hidden)
        gcn_out = gcn_out.view(B, S, N, self.hidden_channels)
        
        # Transpose for LSTM: (Batch, Nodes, SeqLen, Hidden)
        gcn_out = gcn_out.permute(0, 2, 1, 3).contiguous()
        
        # Flatten batch and nodes: (Batch*Nodes, SeqLen, Hidden)
        lstm_in = gcn_out.view(B * N, S, self.hidden_channels)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        
        # Take last hidden state from top layer: (Batch*Nodes, Hidden)
        last_hidden = h_n[-1]
        
        # Project to prediction horizon
        pred_flat = F.relu(self.fc1(last_hidden))
        pred_flat = self.fc_dropout(pred_flat)
        pred_flat = self.fc2(pred_flat) # (Batch*Nodes, PredHorizon * OutChannels)
        
        # Reshape to (Batch, Nodes, PredHorizon, OutChannels)
        pred = pred_flat.view(B, N, self.pred_horizon, self.out_channels)
        
        # Residual skip connection: model predicts delta on top of last observed demand
        # We only apply the demand baseline to the first output feature (Demand)
        demand_baseline = x.permute(0, 2, 1, 3)[:, :, -1, 0]  # (B, N) — last observed demand
        
        demand_pred = pred[:, :, :, 0] + demand_baseline.unsqueeze(-1).expand(-1, -1, self.pred_horizon)
        
        if self.out_channels == 2:
            excess_pred = pred[:, :, :, 1]
            pred_combined = torch.stack([demand_pred, excess_pred], dim=-1)
        else:
            # If out_channels == 1, we just return the demand prediction
            # But the caller might expect (B, N, S, 2). Let's return (B, N, S, 1) and let wrapper handle it
            pred_combined = demand_pred.unsqueeze(-1)
            
        return pred_combined
