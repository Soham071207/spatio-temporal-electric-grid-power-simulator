import os
import json
import csv
import time
import urllib.request
import urllib.error
import re
import calendar
from datetime import datetime
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_demand.log'),
        logging.StreamHandler()
    ]
)

def get_regions():
    # Fallback to known CCAA mapping
    return {
        'Andalucía': '4', 'Aragón': '5', 'Cantabria': '6', 'Castilla la Mancha': '7',
        'Castilla y León': '8', 'Cataluña': '9', 'País Vasco': '10', 'Principado de Asturias': '11',
        'Comunidad de Ceuta': '8744', 'Comunidad de Melilla': '8745', 'Comunidad de Madrid': '13',
        'Comunidad de Navarra': '14', 'Comunidad Valenciana': '15', 'Extremadura': '16',
        'Galicia': '17', 'Islas Baleares': '8743', 'Islas Canarias': '8742', 'La Rioja': '20',
        'Región de Murcia': '21'
    }

def extract_values(data):
    if not isinstance(data, (dict, list)): return []
    results = []
    def search(obj):
        if isinstance(obj, dict):
            if 'datetime' in obj and 'value' in obj:
                results.append(obj)
            else:
                for k, v in obj.items(): search(v)
        elif isinstance(obj, list):
            for item in obj: search(item)
    search(data)
    return results

def fetch_data(url, token):
    retries = 3
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    if token:
        # e-sios and apidatos token header
        headers['x-api-key'] = token
        headers['Authorization'] = f'Token token="{token}"'
        
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            data = resp.read().decode('utf-8')
            return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in [400, 404]: return None
            if attempt == retries - 1:
                logging.error(f"Failed {url} (HTTP {e.code})")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries - 1: return None
            time.sleep(2 ** attempt)
    return None

def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_\-\.]', '_', name)

def main():
    load_dotenv()
    esios_token = os.environ.get("ESIOS_TOKEN")
    if not esios_token or "your_esios_token_here" in esios_token:
        logging.error("Valid ESIOS_TOKEN not found in .env")
        return
        
    start_year = 2015
    end_year = datetime.now().year
    now = datetime.now()
    
    proc_dir = os.path.join("data", "processed", "demand")
    os.makedirs(proc_dir, exist_ok=True)
    
    regions = get_regions()
    # Add national as a fallback
    download_targets = [("National Spain", None)] + list(regions.items())
    
    for name, geo_id in download_targets:
        safe_name = sanitize_filename(name)
        logging.info(f"--- Processing {name} ---")
        
        for year in range(start_year, end_year + 1):
            start_month = 1
            end_month = 12 if year < now.year else now.month
            
            for month in range(start_month, end_month + 1):
                # IDEMPOTENCY CHECK: Do not redownload if file exists
                month_csv = os.path.join(proc_dir, f"{safe_name}_{year}_{month:02d}.csv")
                if os.path.exists(month_csv):
                    logging.info(f"[{name}] {year}-{month:02d} already exists. Skipping.")
                    continue
                
                _, last_day = calendar.monthrange(year, month)
                if year == now.year and month == now.month: last_day = now.day
                    
                start_date = f"{year}-{month:02d}-01T00:00"
                end_date = f"{year}-{month:02d}-{last_day:02d}T23:59"
                
                base_url = f"https://apidatos.ree.es/en/datos/demanda/evolucion?start_date={start_date}&end_date={end_date}"
                if geo_id:
                    base_url += f"&geo_limit=ccaa&geo_ids={geo_id}"
                
                # We attempt hourly. If apidatos rejects hourly, we fall back to daily so we have SOMETHING.
                url = base_url + "&time_trunc=hour"
                data = fetch_data(url, esios_token)
                
                if not data:
                    # Fallback to daily if hourly is blocked for this region
                    url = base_url + "&time_trunc=day"
                    data = fetch_data(url, esios_token)
                    
                values = extract_values(data) if data else []
                
                if values:
                    with open(month_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["region", "datetime", "value"])
                        for item in values:
                            writer.writerow([name, item.get("datetime"), item.get("value")])
                    logging.info(f"[{name}] Downloaded {len(values)} records for {year}-{month:02d}")
                else:
                    logging.warning(f"[{name}] No data returned for {year}-{month:02d}")
                    # Create an empty marker file to prevent retrying a dead endpoint
                    with open(month_csv, 'w') as f: f.write("region,datetime,value\n")
                    
                time.sleep(0.5)

if __name__ == "__main__":
    main()
