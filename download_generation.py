import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for ENTSO-E token
    # Get yours at https://transparency.entsoe.eu/
    entsoe_token = os.environ.get("ENTSOE_TOKEN")
    if not entsoe_token:
        print("ERROR: ENTSOE_TOKEN not found in environment.")
        print("Please add it to your .env file or export it.")
        print("Format: ENTSOE_TOKEN=your_token_here")
        return

    client = EntsoePandasClient(api_key=entsoe_token)
    
    # Spain country code for ENTSO-E
    country_code = 'ES'
    
    # Timeframe (Testing with a single month first to avoid timeouts)
    start = pd.Timestamp('2024-01-01', tz='Europe/Madrid')
    end = pd.Timestamp('2024-02-01', tz='Europe/Madrid')
    
    out_dir = os.path.join("data", "processed", "generation")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Fetching Generation Data for {country_code} from {start} to {end}...")
    
    try:
        # 1. Fetch Aggregated Generation per Production Type
        print("--> Fetching aggregated generation by technology...")
        df_gen = client.query_generation(country_code, start=start, end=end)
        
        # Flatten multi-level columns if they exist (e.g., 'Actual Aggregated' / 'Actual Consumption')
        if isinstance(df_gen.columns, pd.MultiIndex):
            df_gen.columns = ['_'.join(col).strip() for col in df_gen.columns.values]
            
        gen_path = os.path.join(out_dir, 'aggregated_generation_ES_2024_01.csv')
        df_gen.to_csv(gen_path)
        print(f"Saved aggregated generation to {gen_path}")
        
        # 2. Fetch Actual Generation per Generation Unit (Telemetry for Digital Twin)
        # This returns data for individual power plants (Bidding Zone level)
        print("--> Fetching unit-level telemetry (Digital Twin core data)...")
        df_units = client.query_generation_per_plant(country_code, start=start, end=end)
        
        units_path = os.path.join(out_dir, 'unit_generation_ES_2024_01.csv')
        df_units.to_csv(units_path)
        print(f"Saved unit telemetry to {units_path}")
        
        # 3. Fetch Installed Generation Capacity per Production Type
        print("--> Fetching installed capacity by technology...")
        df_cap = client.query_installed_generation_capacity(country_code, start=start, end=end)
        
        cap_path = os.path.join(out_dir, 'installed_capacity_ES_2024.csv')
        df_cap.to_csv(cap_path)
        print(f"Saved installed capacity to {cap_path}")

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        
if __name__ == "__main__":
    main()
