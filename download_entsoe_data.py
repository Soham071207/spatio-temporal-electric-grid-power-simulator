import os
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN")

if not ENTSOE_TOKEN or ENTSOE_TOKEN == 'your_entsoe_token_here':
    logging.error("Valid ENTSOE_TOKEN not found in .env.")
    exit(1)

OUTPUT_DIR = "data/processed/entsoe_generation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_SPAIN = "10YES-REE------0"

def fetch_entsoe_generation(start_dt, end_dt):
    """Fetches Actual Generation per Unit (A73) from ENTSO-E."""
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str = end_dt.strftime("%Y%m%d%H%M")
    
    url = (
        f"https://web-api.tp.entsoe.eu/api?"
        f"securityToken={ENTSOE_TOKEN}&"
        f"documentType=A73&"
        f"processType=A16&"
        f"in_Domain={DOMAIN_SPAIN}&"
        f"periodStart={start_str}&"
        f"periodEnd={end_str}"
    )
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read().decode('utf-8')
                return xml_data
        except Exception as e:
            logging.error(f"Error fetching ENTSO-E data: {e}. Retrying in 5s...")
            time.sleep(5)
            
    return None

def parse_entsoe_xml(xml_data):
    """Parses ENTSO-E XML response into a list of records."""
    if not xml_data:
        return []
        
    try:
        root = ET.fromstring(xml_data)
        records = []
        for ts in root.findall('.//{*}TimeSeries'):
            mrid_elem = ts.find('.//{*}registeredResource.mRID')
            name_elem = ts.find('.//{*}PowerSystemResources/{*}name')
            psr_type_elem = ts.find('.//{*}MktPSRType/{*}psrType')
            
            unit_id = mrid_elem.text if mrid_elem is not None else 'UNKNOWN'
            unit_name = name_elem.text if name_elem is not None else unit_id
            tech_code = psr_type_elem.text if psr_type_elem is not None else 'UNKNOWN'
            
            period = ts.find('.//{*}Period')
            if period is None:
                continue
                
            start_time_elem = period.find('.//{*}timeInterval/{*}start')
            if start_time_elem is None:
                continue
            start_time = pd.to_datetime(start_time_elem.text)
            
            res_elem = period.find('.//{*}resolution')
            resolution = res_elem.text if res_elem is not None else 'PT60M'
            
            for pt in period.findall('.//{*}Point'):
                pos_elem = pt.find('.//{*}position')
                qty_elem = pt.find('.//{*}quantity')
                if pos_elem is None or qty_elem is None:
                    continue
                pos = int(pos_elem.text)
                qty = float(qty_elem.text)
                
                if resolution == 'PT60M':
                    dt = start_time + pd.Timedelta(hours=pos-1)
                elif resolution == 'PT15M':
                    dt = start_time + pd.Timedelta(minutes=(pos-1)*15)
                elif resolution == 'PT5M':
                    dt = start_time + pd.Timedelta(minutes=(pos-1)*5)
                else:
                    dt = start_time
                    
                records.append({
                    'datetime': dt.tz_localize(None),
                    'unit_id': unit_id,
                    'unit_name': unit_name,
                    'tech_code': tech_code,
                    'generation_mw': qty
                })
                
        return records
    except Exception as e:
        logging.error(f"Error parsing XML: {e}")
        return []

def download_11_years():
    logging.info("Starting ENTSO-E plant-level generation download (2015-2025)...")
    output_file = os.path.join(OUTPUT_DIR, "generation_per_plant_11_years.csv")
    
    current_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    
    if end_date > datetime.now():
        end_date = datetime.now()
        
    # Check where we left off to resume
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        if not df_existing.empty:
            last_date_str = df_existing['datetime'].max()
            last_date = pd.to_datetime(last_date_str)
            current_date = datetime(last_date.year, last_date.month, last_date.day) + timedelta(days=1)
            logging.info(f"Resuming download from {current_date.strftime('%Y-%m-%d')}")
            
    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        
        logging.info(f"Fetching ENTSO-E data for {current_date.strftime('%Y-%m-%d')}...")
        xml_data = fetch_entsoe_generation(current_date, next_date)
        records = parse_entsoe_xml(xml_data)
        
        if records:
            df = pd.DataFrame(records)
            df_hourly = df.groupby(['unit_id', 'unit_name', 'tech_code', pd.Grouper(key='datetime', freq='1h')])['generation_mw'].mean().reset_index()
            df_hourly = df_hourly.sort_values(['unit_name', 'datetime'])
            
            # Append to CSV
            df_hourly.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            logging.info(f"Saved {len(df_hourly)} hourly records for {current_date.strftime('%Y-%m-%d')} to {output_file}")
        
        time.sleep(1.5)
        current_date = next_date

if __name__ == "__main__":
    download_11_years()
