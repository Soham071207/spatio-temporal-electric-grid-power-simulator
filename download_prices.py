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

OUTPUT_DIR = "data/processed/prices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_SPAIN = "10YES-REE------0"

def fetch_entsoe_prices(start_dt, end_dt):
    """Fetches Day-Ahead Prices (A44) from ENTSO-E."""
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str = end_dt.strftime("%Y%m%d%H%M")
    
    url = (
        f"https://web-api.tp.entsoe.eu/api?"
        f"securityToken={ENTSOE_TOKEN}&"
        f"documentType=A44&"
        f"in_Domain={DOMAIN_SPAIN}&"
        f"out_Domain={DOMAIN_SPAIN}&"
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
            logging.error(f"Error fetching ENTSO-E price data: {e}. Retrying in 5s...")
            time.sleep(5)
            
    return None

def parse_entsoe_xml(xml_data):
    """Parses ENTSO-E XML response into a list of price records."""
    if not xml_data:
        return []
        
    try:
        root = ET.fromstring(xml_data)
        records = []
        for ts in root.findall('.//{*}TimeSeries'):
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
                price_elem = pt.find('.//{*}price.amount')
                if pos_elem is None or price_elem is None:
                    continue
                pos = int(pos_elem.text)
                price = float(price_elem.text)
                
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
                    'price_eur_mwh': price
                })
                
        return records
    except Exception as e:
        logging.error(f"Error parsing XML: {e}")
        return []

def download_11_years():
    logging.info("Starting ENTSO-E wholesale price download (2015-2026)...")
    output_file = os.path.join(OUTPUT_DIR, "day_ahead_prices.csv")
    
    current_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    
    # Check where we left off to resume
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        if not df_existing.empty:
            last_date_str = df_existing['datetime'].max()
            last_date = pd.to_datetime(last_date_str)
            current_date = datetime(last_date.year, last_date.month, 1) + pd.DateOffset(months=1)
            logging.info(f"Resuming download from {current_date.strftime('%Y-%m-%d')}")
            
    while current_date < end_date:
        next_month = current_date + pd.DateOffset(months=1)
        
        logging.info(f"Fetching ENTSO-E price data for {current_date.strftime('%Y-%m')}...")
        xml_data = fetch_entsoe_prices(current_date, next_month)
        records = parse_entsoe_xml(xml_data)
        
        if records:
            df = pd.DataFrame(records)
            df_hourly = df.groupby(pd.Grouper(key='datetime', freq='1h'))['price_eur_mwh'].mean().reset_index()
            df_hourly = df_hourly.sort_values('datetime')
            
            # Append to CSV
            df_hourly.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            logging.info(f"Saved {len(df_hourly)} hourly records for {current_date.strftime('%Y-%m')} to {output_file}")
        else:
            logging.warning(f"No records returned for {current_date.strftime('%Y-%m')}")
            
        time.sleep(1.5)
        current_date = next_month

if __name__ == "__main__":
    download_11_years()
