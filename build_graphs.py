import os
import numpy as np
import pandas as pd

# 1. Node definitions and coordinates (lat, lon)
REGIONS = {
    'Andalucía': (37.3828, -5.9732),
    'Aragón': (41.6561, -0.8773),
    'Cantabria': (43.4647, -3.8044),
    'Castilla la Mancha': (39.8628, -4.0273),
    'Castilla y León': (41.6523, -4.7245),
    'Cataluña': (41.3888, 2.1590),
    'País Vasco': (42.8590, -2.6818),
    'Principado de Asturias': (43.3614, -5.8593),
    'Comunidad de Ceuta': (35.8894, -5.3213),
    'Comunidad de Melilla': (35.2923, -2.9381),
    'Comunidad de Madrid': (40.4165, -3.7026),
    'Comunidad de Navarra': (42.8169, -1.6432),
    'Comunidad Valenciana': (39.4697, -0.3774),
    'Extremadura': (38.9161, -6.3437),
    'Galicia': (42.8805, -8.5457),
    'Islas Baleares': (39.5694, 2.6502),
    'Islas Canarias': (28.1235, -15.4363),
    'La Rioja': (42.4627, -2.4450),
    'Región de Murcia': (37.9870, -1.1300)
}

# 2. Electrical interconnections proxy (shared borders + known HVDC links)
BORDERS = {
    'Andalucía': ['Extremadura', 'Castilla la Mancha', 'Región de Murcia'],
    'Aragón': ['Cataluña', 'Comunidad Valenciana', 'Castilla la Mancha', 'Castilla y León', 'La Rioja', 'Comunidad de Navarra'],
    'Cantabria': ['Principado de Asturias', 'Castilla y León', 'País Vasco'],
    'Castilla la Mancha': ['Andalucía', 'Extremadura', 'Castilla y León', 'Comunidad de Madrid', 'Aragón', 'Comunidad Valenciana', 'Región de Murcia'],
    'Castilla y León': ['Galicia', 'Principado de Asturias', 'Cantabria', 'País Vasco', 'La Rioja', 'Aragón', 'Castilla la Mancha', 'Comunidad de Madrid', 'Extremadura'],
    'Cataluña': ['Aragón', 'Comunidad Valenciana'],
    'Comunidad de Madrid': ['Castilla y León', 'Castilla la Mancha'],
    'Comunidad de Navarra': ['País Vasco', 'La Rioja', 'Aragón'],
    'Comunidad Valenciana': ['Cataluña', 'Aragón', 'Castilla la Mancha', 'Región de Murcia', 'Islas Baleares'], # Added Romulo HVDC link
    'Extremadura': ['Andalucía', 'Castilla la Mancha', 'Castilla y León'],
    'Galicia': ['Principado de Asturias', 'Castilla y León'],
    'Islas Baleares': ['Comunidad Valenciana'], # Added Romulo HVDC link
    'Islas Canarias': [], # Isolated
    'La Rioja': ['Castilla y León', 'País Vasco', 'Comunidad de Navarra', 'Aragón'],
    'País Vasco': ['Cantabria', 'Castilla y León', 'La Rioja', 'Comunidad de Navarra'],
    'Principado de Asturias': ['Galicia', 'Cantabria', 'Castilla y León'],
    'Región de Murcia': ['Andalucía', 'Castilla la Mancha', 'Comunidad Valenciana'],
    'Comunidad de Ceuta': ['Andalucía'], # AC submarine cable REMEDIO (Tarifa-Fardioa) exists
    'Comunidad de Melilla': [] # Isolated
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees) in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def compute_A_geo(regions, coords, sigma=100.0, threshold=0.1):
    """Computes Gaussian kernel geographic adjacency matrix."""
    N = len(regions)
    A = np.zeros((N, N))
    
    # Compute pairwise distances
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = haversine(coords[regions[i]][0], coords[regions[i]][1], 
                                          coords[regions[j]][0], coords[regions[j]][1])
            
    # Apply Gaussian kernel to convert distance to weight: w_ij = exp(-(d_ij^2) / sigma^2)
    # sigma controls how fast the weight decays with distance
    for i in range(N):
        for j in range(N):
            if i != j:
                weight = np.exp(-(dist_matrix[i, j]**2) / (sigma**2))
                A[i, j] = weight if weight >= threshold else 0.0
            else:
                A[i, j] = 1.0 # Self-loop
                
    return A

def compute_A_elec(regions, borders):
    """Computes binary electrical adjacency matrix based on physical connections."""
    N = len(regions)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i, j] = 1.0 # Self-loop
            elif regions[j] in borders[regions[i]]:
                A[i, j] = 1.0
                
    return A

def main():
    out_dir = os.path.join("data", "processed", "graphs")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save Node Mapping
    regions_list = list(REGIONS.keys())
    node_df = pd.DataFrame({
        'node_id': range(len(regions_list)),
        'region': regions_list,
        'latitude': [REGIONS[r][0] for r in regions_list],
        'longitude': [REGIONS[r][1] for r in regions_list]
    })
    node_df.to_csv(os.path.join(out_dir, 'node_ids.csv'), index=False)
    print(f"Saved node_ids.csv with {len(regions_list)} regions.")
    
    # 2. Compute and Save Geographical Graph (A_geo)
    # using sigma=200km to capture strong correlations within that radius
    A_geo = compute_A_geo(regions_list, REGIONS, sigma=200.0, threshold=0.1) 
    np.save(os.path.join(out_dir, 'A_geo.npy'), A_geo)
    print(f"Saved A_geo.npy with shape {A_geo.shape} and {np.count_nonzero(A_geo)} edges.")
    
    # 3. Compute and Save Electrical Graph (A_elec)
    A_elec = compute_A_elec(regions_list, BORDERS)
    np.save(os.path.join(out_dir, 'A_elec.npy'), A_elec)
    print(f"Saved A_elec.npy with shape {A_elec.shape} and {np.count_nonzero(A_elec)} edges.")

if __name__ == "__main__":
    main()
