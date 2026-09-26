import os
import urllib.request
import json

OUTPUT_FILE = "data/processed/graphs/spain_ccaa.geojson"

def download_geojson():
    # URL to a high-quality open-source GeoJSON of Spanish Autonomous Communities
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-communities.geojson"
    
    print(f"Downloading Spanish CCAA GeoJSON from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
        print(f"GeoJSON downloaded and saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error downloading GeoJSON: {e}")

if __name__ == "__main__":
    os.makedirs("data/processed/graphs", exist_ok=True)
    download_geojson()
