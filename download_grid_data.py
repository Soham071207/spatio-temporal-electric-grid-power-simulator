import os
import urllib.request
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_grid.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

load_dotenv()
ESIOS_TOKEN = os.getenv("ESIOS_TOKEN")

if not ESIOS_TOKEN:
    logging.error("ESIOS_TOKEN not found in .env. Exiting.")
    exit(1)

HEADERS = {
    'Accept': 'application/json; application/vnd.esios-api-v1+json',
    'Content-Type': 'application/json',
    'x-api-key': ESIOS_TOKEN
}

# The indicators we need
INDICATORS = {
    # Supply-side 5-min generation
    'gen_hydro': 546,
    'gen_coal': 547,
    'gen_nuclear': 549,
    'gen_ccgt': 550,
    'gen_wind': 551,
    'gen_solar': 552,
    'gen_solar_thermal': 1294,
    'gen_solar_pv': 1295,
    'interchanges': 553,
    
    # Demand
    'demand_national': 1293,  # 5-min
    'demand_baleares': 10244, # Hourly
    'demand_canarias': 10243, # Hourly
    'demand_ccaa_monthly': 10339 # Monthly
}

OUTPUT_DIR = "data/processed/grid_telemetry"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_indicator_month(indicator_id, start_date, end_date):
    """Fetches data for a specific indicator and month with exponential backoff."""
    url = f"https://api.esios.ree.es/indicators/{indicator_id}?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
    
    max_retries = 5
    base_wait = 2
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                if 'indicator' not in data or 'values' not in data['indicator']:
                    logging.warning(f"No 'values' found for indicator {indicator_id} in {start_date.strftime('%Y-%m')}")
                    return []
                    
                return data['indicator']['values']
                
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait_time = base_wait * (2 ** attempt)
                logging.warning(f"HTTP {e.code} for indicator {indicator_id}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTP {e.code} for indicator {indicator_id}: {e.read().decode('utf-8')}")
                return []
        except Exception as e:
            wait_time = base_wait * (2 ** attempt)
            logging.error(f"Error for indicator {indicator_id}: {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    logging.error(f"Failed to fetch indicator {indicator_id} for {start_date.strftime('%Y-%m')} after {max_retries} attempts.")
    return []

def process_and_save(name, indicator_id, start_year=2015, end_year=2025):
    """Downloads all months for an indicator, processes to hourly, and saves to CSV."""
    logging.info(f"--- Starting download for {name} (ID: {indicator_id}) ---")
    
    output_file = os.path.join(OUTPUT_DIR, f"{name}.csv")
    
    # Check if already completed
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            if not df_existing.empty:
                last_date_str = df_existing['datetime'].max()
                last_date = pd.to_datetime(last_date_str)
                if last_date.year >= end_year and last_date.month == 12:
                    logging.info(f"{name} already completely downloaded. Skipping.")
                    return
        except Exception as e:
            logging.warning(f"Failed to read existing file {output_file}. Overwriting. ({e})")
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Special case to avoid querying the future
            if year == datetime.now().year and month > datetime.now().month:
                break
                
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
                
            logging.info(f"Fetching {name} for {start_date.strftime('%Y-%m')}")
            
            raw_values = fetch_indicator_month(indicator_id, start_date, end_date)
            
            if not raw_values:
                continue
                
            df_month = pd.DataFrame(raw_values)
            
            if df_month.empty:
                continue
                
            if 'datetime_utc' in df_month.columns:
                df_month['datetime'] = pd.to_datetime(df_month['datetime_utc']).dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                df_month['datetime'] = pd.to_datetime(df_month['datetime'], utc=True).dt.tz_localize(None)
                
            if 'geo_id' in df_month.columns and df_month['geo_id'].nunique() > 1:
                df_resampled = df_month.set_index('datetime').groupby('geo_id').resample('1h')['value'].mean().reset_index()
                geo_names = df_month[['geo_id', 'geo_name']].drop_duplicates()
                df_resampled = df_resampled.merge(geo_names, on='geo_id', how='left')
            else:
                if indicator_id == 10339:
                    df_resampled = df_month[['datetime', 'value', 'geo_id', 'geo_name']]
                else:
                    df_resampled = df_month.set_index('datetime').resample('1h')['value'].mean().reset_index()
                    
            all_data.append(df_resampled)
            time.sleep(1)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        if 'geo_id' in final_df.columns:
            final_df = final_df.sort_values(['geo_id', 'datetime'])
        else:
            final_df = final_df.sort_values('datetime')
            
        final_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(final_df)} records to {output_file}")
    else:
        logging.warning(f"No data downloaded for {name}")

if __name__ == "__main__":
    logging.info("Starting grid data download pipeline for ALL years...")
    
    # We are pulling the full 11-year dataset (2015-2025) as requested: NO CORNERS CUT.
    start_y = 2015
    end_y = 2025
    
    for name, ind_id in INDICATORS.items():
        process_and_save(name, ind_id, start_year=start_y, end_year=end_y)
        
    logging.info("Pipeline complete.")
