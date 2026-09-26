import os
import json
import random
import pandas as pd

# Approximate centers of Spanish regions [latitude, longitude]
REGION_COORDS = {
    "Andalucía": [37.3828, -5.9731],
    "Aragón": [41.6561, -0.8773],
    "Principado de Asturias": [43.3614, -5.8593],
    "Illes Balears": [39.6953, 3.0176],
    "Canarias": [28.2916, -16.6291],
    "Cantabria": [43.1828, -3.9878],
    "Castilla y León": [41.6523, -4.7245],
    "Castilla-La Mancha": [39.8628, -4.0273],
    "Cataluña": [41.3851, 2.1734],
    "Comunitat Valenciana": [39.4699, -0.3774],
    "Extremadura": [39.4806, -6.3722],
    "Galicia": [42.8782, -8.5448],
    "Comunidad de Madrid": [40.4168, -3.7038],
    "Región de Murcia": [37.9922, -1.1307],
    "Comunidad Foral de Navarra": [42.8125, -1.6458],
    "País Vasco": [43.2627, -2.9253],
    "La Rioja": [42.4627, -2.4450],
    "Ceuta": [35.8894, -5.3213],
    "Melilla": [35.2923, -2.9381],
}

def generate_mock_coords():
    # Read the metadata
    project_root = os.path.dirname(os.path.dirname(__file__))
    metadata_path = os.path.join(project_root, "data", "processed", "plant_metadata.csv")
    
    if not os.path.exists(metadata_path):
        print("Metadata not found, make sure to run build_plant_metadata.py first.")
        return
        
    df = pd.read_csv(metadata_path)
    
    plants = []
    
    random.seed(42) # Deterministic jitter
    
    for _, row in df.iterrows():
        region = row['region']
        # Handle some slight naming mismatches between metadata and REGION_COORDS
        if region == "Comunidad Valenciana": region = "Comunitat Valenciana"
        if region not in REGION_COORDS:
            region = "Comunidad de Madrid" # fallback
            
        base_lat, base_lon = REGION_COORDS[region]
        
        # Add a random jitter (approx +/- 0.5 degrees, roughly 50km)
        lat = base_lat + random.uniform(-0.6, 0.6)
        lon = base_lon + random.uniform(-0.6, 0.6)
        
        plants.append({
            "id": row['unit_id'],
            "name": row['unit_name'],
            "tech": row['technology'],
            "category": row['tech_category'],
            "region": row['region'],
            "capacity_mw": row['estimated_capacity_mw'],
            "lat": lat,
            "lon": lon
        })
        
    out_dir = os.path.join(project_root, "frontend", "src", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "plant_coords.json")
    
    with open(out_path, 'w') as f:
        json.dump(plants, f, indent=2)
        
    print(f"Generated {len(plants)} plant coordinates at {out_path}")

if __name__ == "__main__":
    generate_mock_coords()
