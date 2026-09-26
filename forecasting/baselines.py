import torch
import torch.nn as nn

class PersistenceModel:
    """
    Baseline model that predicts the last observed value will persist for all future steps.
    """
    def __init__(self, pred_horizon):
        self.pred_horizon = pred_horizon
        
    def predict(self, x):
        """
        x: (Batch, Nodes, SeqLen, Features)
        returns: (Batch, Nodes, PredHorizon)
        """
        # Take the last time step from the sequence (index -1)
        last_obs = x[:, :, -1, 0] # Shape: (Batch, Nodes)
        
        # Repeat it for the prediction horizon
        # Target shape: (Batch, Nodes, PredHorizon)
        pred = last_obs.unsqueeze(-1).repeat(1, 1, self.pred_horizon)
        return pred

class LSTMModel(nn.Module):
    """
    Baseline LSTM model that processes each node's sequence independently.
    No spatial/graph information is used.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, pred_horizon):
        super(LSTMModel, self).__init__()
        self.pred_horizon = pred_horizon
        
        # LSTM processes sequences. Input: (Batch*Nodes, SeqLen, Features)
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        self.fc = nn.Linear(hidden_channels, pred_horizon)
        
    def forward(self, x):
        """
        x: (Batch, Nodes, SeqLen, Features)
        returns: (Batch, Nodes, PredHorizon)
        """
        B, N, S, F = x.shape
        # Flatten batch and nodes to process each node's sequence independently
        x_flat = x.view(B * N, S, F)
        
        # LSTM output: out is (Batch*Nodes, SeqLen, Hidden), (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x_flat)
        
        # Take the last hidden state
        last_hidden = h_n[-1] # (Batch*Nodes, Hidden)
        
        # Project to prediction horizon
        pred_flat = self.fc(last_hidden) # (Batch*Nodes, PredHorizon)
        
        # Reshape back to (Batch, Nodes, PredHorizon)
        pred = pred_flat.view(B, N, self.pred_horizon)
        
        return pred
