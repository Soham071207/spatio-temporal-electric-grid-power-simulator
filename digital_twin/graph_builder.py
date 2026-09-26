import os
import pandas as pd
import numpy as np
import torch
# Make PyG optional during build/test if not installed yet
try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging = __import__("logging")
    logging.warning("torch_geometric is not installed. PyG Data objects cannot be created.")

class GridGraphBuilder:
    def __init__(self, data_dir="data/processed/grid_telemetry", entsoe_dir="data/processed/entsoe_generation"):
        self.data_dir = data_dir
        self.entsoe_dir = entsoe_dir
        
    def load_snapshot(self, target_time: pd.Timestamp):
        """Loads data for a specific hour and constructs the graph nodes/edges."""
        # 1. Load Regional Demand Nodes (19 nodes)
        regional_demand_path = os.path.join(self.data_dir, "demand_regional_disaggregated.csv")
        df_demand = pd.read_csv(regional_demand_path)
        df_demand['datetime'] = pd.to_datetime(df_demand['datetime'])
        current_demand = df_demand[df_demand['datetime'] == target_time]
        
        # 2. Load ENTSO-E Plant Generation Nodes (~170 nodes)
        entsoe_path = os.path.join(self.entsoe_dir, "generation_per_plant_11_years.csv")
        df_gen = pd.DataFrame() # Fallback
        if os.path.exists(entsoe_path):
            df_gen = pd.read_csv(entsoe_path)
            df_gen['datetime'] = pd.to_datetime(df_gen['datetime'])
            df_gen = df_gen[df_gen['datetime'] == target_time]
            
        # 3. Load Interconnect Nodes (4 nodes)
        interconnect_path = os.path.join(self.data_dir, "interchanges.csv")
        df_inter = pd.read_csv(interconnect_path)
        df_inter['datetime'] = pd.to_datetime(df_inter['datetime'])
        current_inter = df_inter[df_inter['datetime'] == target_time]
        
        return self._build_pyg_data(current_demand, df_gen, current_inter)
        
    def _build_pyg_data(self, demand_df, gen_df, inter_df):
        if not PYG_AVAILABLE:
            raise ImportError("Please pip install torch_geometric")
            
        # Node Indexing: 
        # 0-18: Regional Demand Nodes
        # 19 to 19+N: Generation Plant Nodes
        # End: Interconnects
        
        x = []
        node_types = [] # 0: Demand, 1: Generator, 2: Interconnect
        
        # Add Demand Nodes
        for _, row in demand_df.iterrows():
            # Features: [Demand_MW, 0, 0]
            x.append([row['demand_mwh'], 0.0, 0.0])
            node_types.append(0)
            
        # Add Generation Nodes
        for _, row in gen_df.iterrows():
            # Features: [0, Generation_MW, 0]
            x.append([0.0, row['generation_mw'], 0.0])
            node_types.append(1)
            
        # We assume a fully connected bipartite graph between Generation and Demand for the MVP,
        # or we can use distance-based edges if we map lat/lon.
        # For a chaotic Digital Twin, we want power to flow from all generators to all regions,
        # weighted by distance, but for MVP a fully connected graph works.
        num_demand = len(demand_df)
        num_gen = len(gen_df)
        
        edge_index = []
        edge_attr = []
        
        # Connect every generator to every demand node
        for gen_idx in range(num_demand, num_demand + num_gen):
            for dem_idx in range(num_demand):
                edge_index.append([gen_idx, dem_idx])
                edge_index.append([dem_idx, gen_idx]) # Bidirectional
                # Edge weights could be transmission line capacity, default to 1.0
                edge_attr.append([1.0])
                edge_attr.append([1.0])
                
        x_tensor = torch.tensor(x, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, node_type=node_types_tensor)
        return data

if __name__ == "__main__":
    print("Graph Builder module ready.")
