import os
import pandas as pd
import json
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

INPUT_FILE = "data/processed/entsoe_generation/generation_per_plant_11_years.csv"
OUTPUT_FILE = "data/processed/graphs/plant_locations.json"

def get_unique_plants():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return pd.DataFrame()
        
    print("Reading ENTSO-E data to extract unique power plants...")
    # Read chunk by chunk to avoid loading entire 11 years into memory
    chunks = pd.read_csv(INPUT_FILE, chunksize=100000, usecols=['unit_id', 'unit_name', 'tech_code'])
    unique_plants = pd.DataFrame()
    
    for chunk in chunks:
        unique_plants = pd.concat([unique_plants, chunk.drop_duplicates()])
        unique_plants = unique_plants.drop_duplicates(subset=['unit_id'])
        
    print(f"Found {len(unique_plants)} unique power plants.")
    return unique_plants

def geocode_plants(df):
    geolocator = Nominatim(user_agent="spain_digital_twin_geocoder")
    results = {}
    
    # Load existing to resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
    # Additional hints to help geocoder find Spanish plants
    hints = " Spain power plant"
    
    count = 0
    for _, row in df.iterrows():
        uid = row['unit_id']
        name = row['unit_name']
        tech = row['tech_code']
        
        if uid in results:
            continue
            
        search_query = f"{name}{hints}"
        print(f"Geocoding: {search_query}")
        
        try:
            location = geolocator.geocode(search_query, timeout=10)
            if location:
                results[uid] = {
                    "name": name,
                    "tech_code": tech,
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "address": location.address
                }
                print(f"  -> Found: {location.latitude}, {location.longitude}")
            else:
                # Try without "power plant" string
                alt_query = f"{name} Spain"
                location = geolocator.geocode(alt_query, timeout=10)
                if location:
                    results[uid] = {
                        "name": name,
                        "tech_code": tech,
                        "lat": location.latitude,
                        "lon": location.longitude,
                        "address": location.address
                    }
                    print(f"  -> Found (Alt): {location.latitude}, {location.longitude}")
                else:
                    print(f"  -> Not found.")
                    # Insert null to avoid retrying endlessly
                    results[uid] = {
                        "name": name,
                        "tech_code": tech,
                        "lat": None,
                        "lon": None,
                        "address": None
                    }
                    
            # Save every 5 plants to avoid losing progress
            count += 1
            if count % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                    
            time.sleep(1.5) # Nominatim rate limit is 1 req/sec
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"  -> API Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            
    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("Geocoding complete!")
    
if __name__ == "__main__":
    os.makedirs("data/processed/graphs", exist_ok=True)
    df = get_unique_plants()
    if not df.empty:
        geocode_plants(df)
