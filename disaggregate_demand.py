import os
import pandas as pd
import numpy as np
import logging
from glob import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base populations for weighting
POPULATION = {
    'Andalucía': 8500000,
    'Cataluña': 7800000,
    'Comunidad de Madrid': 6800000,
    'Comunidad Valenciana': 5100000,
    'Galicia': 2700000,
    'Castilla y León': 2400000,
    'País Vasco': 2200000,
    'Canarias': 2200000,
    'Castilla-La Mancha': 2100000,
    'Región de Murcia': 1500000,
    'Aragón': 1300000,
    'Islas Baleares': 1200000,
    'Extremadura': 1050000,
    'Principado de Asturias': 1000000,
    'Comunidad Foral de Navarra': 660000,
    'Cantabria': 580000,
    'La Rioja': 320000,
    'Melilla': 85000,
    'Ceuta': 84000
}

# Regional demand shape profiles
# Each profile modulates the national hourly curve to create a unique shape
# Values are hourly multipliers (24 hours) relative to 1.0
# Industrial: flat daytime plateau (factories run steadily 7am-7pm)
# Urban: sharp morning + evening peaks (commuter + residential)
# Tourist/Coastal: lower morning, strong evening + night (hospitality, nightlife)
# Rural/Agricultural: early morning activity, lower evening
DEMAND_PROFILES = {
    'industrial': np.array([
        0.70, 0.65, 0.62, 0.60, 0.63, 0.72, 0.88, 1.05,
        1.12, 1.14, 1.15, 1.13, 1.08, 1.12, 1.14, 1.13,
        1.10, 1.05, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75
    ]),
    'urban': np.array([
        0.62, 0.58, 0.55, 0.53, 0.55, 0.65, 0.85, 1.10,
        1.22, 1.18, 1.12, 1.08, 1.05, 1.08, 1.10, 1.12,
        1.15, 1.20, 1.25, 1.22, 1.15, 1.02, 0.88, 0.72
    ]),
    'tourist': np.array([
        0.75, 0.72, 0.68, 0.65, 0.63, 0.65, 0.72, 0.85,
        0.95, 1.00, 1.05, 1.08, 1.10, 1.05, 1.00, 0.98,
        1.02, 1.10, 1.20, 1.28, 1.30, 1.25, 1.10, 0.90
    ]),
    'rural': np.array([
        0.68, 0.63, 0.60, 0.58, 0.62, 0.75, 0.92, 1.10,
        1.18, 1.20, 1.18, 1.12, 1.05, 1.08, 1.10, 1.08,
        1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70
    ]),
    'island': np.array([
        0.78, 0.73, 0.68, 0.65, 0.63, 0.65, 0.75, 0.88,
        0.98, 1.05, 1.10, 1.12, 1.08, 1.02, 0.98, 1.00,
        1.05, 1.15, 1.25, 1.30, 1.28, 1.18, 1.02, 0.85
    ]),
}

# Assign each region its demand profile type
REGION_PROFILE = {
    'Andalucía': 'tourist',
    'Cataluña': 'urban',
    'Comunidad de Madrid': 'urban',
    'Comunidad Valenciana': 'tourist',
    'Galicia': 'rural',
    'Castilla y León': 'rural',
    'País Vasco': 'industrial',
    'Canarias': 'island',
    'Castilla-La Mancha': 'rural',
    'Región de Murcia': 'tourist',
    'Aragón': 'industrial',
    'Islas Baleares': 'island',
    'Extremadura': 'rural',
    'Principado de Asturias': 'industrial',
    'Comunidad Foral de Navarra': 'industrial',
    'Cantabria': 'rural',
    'La Rioja': 'industrial',
    'Melilla': 'island',
    'Ceuta': 'island'
}

# Weekend damping factors per profile type
# Industrial regions drop more on weekends (factories closed)
# Tourist regions drop less (tourism continues)
WEEKEND_DAMPING = {
    'industrial': 0.72,
    'urban': 0.82,
    'tourist': 0.92,
    'rural': 0.80,
    'island': 0.90,
}

# The names in e-sios might have encoding artifacts depending on OS/download.
# We'll map them carefully.
ESIOS_TO_STD = {
    'Andaluca': 'Andalucía',
    'Aragn': 'Aragón',
    'Cantabria': 'Cantabria',
    'Castilla y Len': 'Castilla y León',
    'Castilla-La Mancha': 'Castilla-La Mancha',
    'Catalua': 'Cataluña',
    'Ceuta': 'Ceuta',
    'Comunidad Foral de Navarra': 'Comunidad Foral de Navarra',
    'Comunidad Valenciana': 'Comunidad Valenciana',
    'Comunidad de Madrid': 'Comunidad de Madrid',
    'Extremadura': 'Extremadura',
    'Galicia': 'Galicia',
    'Islas Baleares': 'Islas Baleares',
    'Islas Canarias': 'Canarias',
    'La Rioja': 'La Rioja',
    'Melilla': 'Melilla',
    'Pas Vasco': 'País Vasco',
    'Principado de Asturias': 'Principado de Asturias',
    'Regin de Murcia': 'Región de Murcia'
}

# Weather file name mapping for regions whose demand CSV name
# doesn't match the weather filename
WEATHER_FILE_MAP = {
    'Andalucía': 'Andalucia',
    'Aragón': 'Aragon',
    'Cataluña': 'Cataluna',
    'País Vasco': 'Pais_Vasco',
    'Castilla y León': 'Castilla_y_Leon',
    'Castilla-La Mancha': 'Castilla_la_Mancha',
    'Comunidad de Madrid': 'Comunidad_de_Madrid',
    'Comunidad Valenciana': 'Comunidad_Valenciana',
    'Comunidad Foral de Navarra': 'Comunidad_de_Navarra',
    'Principado de Asturias': 'Principado_de_Asturias',
    'Región de Murcia': 'Region_de_Murcia',
    'Islas Baleares': 'Islas_Baleares',
    'Canarias': 'Islas_Canarias',
    'Ceuta': 'Comunidad_de_Ceuta',
    'Melilla': 'Comunidad_de_Melilla',
    'Extremadura': 'Extremadura',
    'Galicia': 'Galicia',
    'Cantabria': 'Cantabria',
    'La Rioja': 'La_Rioja',
}


def map_esios_name(x):
    if not isinstance(x, str): return x
    mapping = {
        "Andaluc": "Andalucía",
        "Arag": "Aragón",
        "Castilla y Le": "Castilla y León",
        "Catalu": "Cataluña",
        "Canarias": "Canarias",
        "Vasco": "País Vasco",
        "Murcia": "Región de Murcia"
    }
    for k, v in mapping.items():
        if k in x:
            return v
    return ESIOS_TO_STD.get(x, x)

def load_weather_data(weather_dir="data/processed/weather"):
    """Loads and returns weather data as a dictionary keyed by standard region name."""
    weather_data = {}
    files = glob(os.path.join(weather_dir, "*.csv"))
    
    # Build reverse mapping: filename (without .csv) -> standard region name
    filename_to_region = {}
    for region, filename in WEATHER_FILE_MAP.items():
        filename_to_region[filename] = region
    
    for f in files:
        basename = os.path.basename(f).replace(".csv", "")
        region = filename_to_region.get(basename, basename)
        
        df = pd.read_csv(f)
        # Weather CSVs have 'datetime' column, not 'time'
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df = df.set_index('datetime')
            weather_data[region] = df
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df = df.set_index('datetime')
            weather_data[region] = df
            
    logging.info(f"Loaded weather data for {len(weather_data)} regions: {list(weather_data.keys())}")
    return weather_data


def compute_hourly_weights(df_weather, region, national_datetime):
    """
    Computes dynamic hourly weights with STRONG weather sensitivity
    plus regional demand shape profiles.
    
    Weather factors are 10x stronger than before to create genuinely
    different demand curves per region:
    - HDD: 50% demand increase per 10°C below 15°C (heating)
    - CDD: 80% demand increase per 10°C above 24°C (cooling)
    - Solar: 15% demand reduction at peak solar (lighting offset)
    
    The regional demand profile reshapes the hourly curve based on
    the economic character of each region (industrial vs tourist vs urban etc).
    """
    base_weight = POPULATION.get(region, 100000)
    profile_type = REGION_PROFILE.get(region, 'urban')
    profile = DEMAND_PROFILES[profile_type]
    
    # Create hour-of-day profile for the entire datetime index
    hours = national_datetime.hour
    hourly_profile = np.array([profile[h] for h in hours])
    
    # Weekend damping
    is_weekend = national_datetime.dayofweek >= 5
    weekend_factor = np.where(is_weekend, WEEKEND_DAMPING[profile_type], 1.0)
    
    # If we don't have weather data, use profile-only weights
    if region not in df_weather:
        logging.warning(f"No weather data found for '{region}'. Using profile-only weight.")
        return pd.Series(base_weight * hourly_profile * weekend_factor, index=national_datetime)
        
    wdf = df_weather[region].reindex(national_datetime)
    
    # Heating Degree Deviation — STRONG: 50% increase per 10°C below 15
    hdd = np.maximum(0, 15 - wdf['temperature_2m'].fillna(15))
    hdd_factor = (hdd / 10.0) * 0.50
    
    # Cooling Degree Deviation — STRONG: 80% increase per 10°C above 24
    # This creates big afternoon spikes in hot regions like Andalucía/Murcia
    cdd = np.maximum(0, wdf['temperature_2m'].fillna(24) - 24)
    cdd_factor = (cdd / 10.0) * 0.80
    
    # Solar effect — MODERATE: 15% reduction at peak solar
    # Reduces daytime demand in sunny regions, creating shape differences
    solar = wdf['shortwave_radiation'].fillna(0)
    solar_factor = (solar / 1000.0) * 0.15
    
    # Wind chill effect — cold windy days increase heating demand more
    wind = wdf['wind_speed_10m'].fillna(0) if 'wind_speed_10m' in wdf.columns else pd.Series(0, index=national_datetime)
    wind_chill_boost = np.where(
        wdf['temperature_2m'].fillna(15) < 10,
        (wind / 30.0) * 0.10,  # Up to 10% extra demand in cold+windy conditions
        0.0
    )
    
    # Combine all weather factors
    weather_multiplier = (1 + hdd_factor + cdd_factor + wind_chill_boost) * (1 - solar_factor)
    
    # Final weight = population × hourly profile × weather × weekend
    dynamic_weights = base_weight * hourly_profile * weather_multiplier * weekend_factor
    
    return pd.Series(dynamic_weights.values if hasattr(dynamic_weights, 'values') else dynamic_weights, 
                     index=national_datetime)


def disaggregate():
    logging.info("Starting demand disaggregation (v2 — strong weather + profiles)...")
    
    telemetry_dir = "data/processed/grid_telemetry"
    national_file = os.path.join(telemetry_dir, "demand_national.csv")
    monthly_file = os.path.join(telemetry_dir, "demand_ccaa_monthly.csv")
    
    if not os.path.exists(national_file) or not os.path.exists(monthly_file):
        logging.error("Missing telemetry files. Please run download_grid_data.py first.")
        return
        
    df_nat = pd.read_csv(national_file)
    df_nat['datetime'] = pd.to_datetime(df_nat['datetime'])
    df_nat = df_nat.set_index('datetime').sort_index()
    
    df_mon = pd.read_csv(monthly_file)
    df_mon['datetime'] = pd.to_datetime(df_mon['datetime'])
    
    weather_data = load_weather_data()
    
    # Map raw e-sios geo names to our standard ones
    if 'geo_name' in df_mon.columns:
        df_mon['region'] = df_mon['geo_name'].apply(map_esios_name)
    
    disaggregated_dfs = []
    
    regions = df_mon['region'].dropna().unique()
    for region in regions:
        logging.info(f"Disaggregating {region}...")
        
        region_monthly = df_mon[df_mon['region'] == region]
        
        # We will build it month by month to apply the volume constraint
        for _, row in region_monthly.iterrows():
            dt_raw = row['datetime']
            # e-sios monthly data often stamps at the end of previous month or weird UTC offsets
            # e.g., '2023-12-31 23:00:00' represents January 2024. 
            # So if day is > 15, it's actually representing the next month.
            target_year = dt_raw.year
            target_month = dt_raw.month
            if dt_raw.day > 15:
                if target_month == 12:
                    target_month = 1
                    target_year += 1
                else:
                    target_month += 1
                    
            month_start = datetime(target_year, target_month, 1)
            
            if target_month == 12:
                month_end = datetime(target_year + 1, 1, 1) - pd.Timedelta(hours=1)
            else:
                month_end = datetime(target_year, target_month + 1, 1) - pd.Timedelta(hours=1)
                
            # Get national hourly curve for this month
            nat_mask = (df_nat.index >= month_start) & (df_nat.index <= month_end)
            nat_month = df_nat.loc[nat_mask].copy()
            
            if nat_month.empty:
                continue
                
            target_volume = row['value'] # MWh for the month
            
            # Compute dynamic weights for every hour
            hourly_weights = compute_hourly_weights(weather_data, region, nat_month.index)
            
            # Apply weights to national shape
            # shape = nat_value * weight
            raw_hourly = nat_month['value'] * hourly_weights
            
            # Rescale to match exact monthly volume
            current_volume = raw_hourly.sum()
            if current_volume > 0:
                scale_factor = target_volume / current_volume
                final_hourly = raw_hourly * scale_factor
            else:
                final_hourly = raw_hourly
                
            df_res = pd.DataFrame({
                'datetime': final_hourly.index,
                'region': region,
                'demand_mwh': final_hourly.values
            })
            
            disaggregated_dfs.append(df_res)
            
    if disaggregated_dfs:
        final_df = pd.concat(disaggregated_dfs, ignore_index=True)
        final_df = final_df.sort_values(['region', 'datetime'])
        
        output_path = os.path.join(telemetry_dir, "demand_regional_disaggregated.csv")
        final_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved {len(final_df)} disaggregated records to {output_path}")
        
        # Validation: check inter-region correlation
        sample_date = '2025-07-20'
        sample = final_df[final_df['datetime'].astype(str).str.startswith(sample_date)]
        if len(sample) > 0:
            pivot = sample.pivot(index='datetime', columns='region', values='demand_mwh')
            if len(pivot.columns) >= 2:
                normed = pivot.apply(lambda c: (c - c.min()) / max(c.max() - c.min(), 1e-9), axis=0)
                corr_matrix = normed.corr().values
                off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                mean_corr = off_diag.mean()
                logging.info(f"VALIDATION: Mean inter-region normalized correlation on {sample_date}: {mean_corr:.4f}")
                if mean_corr > 0.95:
                    logging.warning("⚠ Curves are still too similar! Check weather sensitivity.")
                else:
                    logging.info("✓ Good diversity — regions have distinct demand shapes!")
    else:
        logging.error("Failed to produce disaggregated data.")

if __name__ == "__main__":
    disaggregate()
