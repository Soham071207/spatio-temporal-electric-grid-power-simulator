import os
import csv
import urllib.request
import urllib.error
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_weather.log'),
        logging.StreamHandler()
    ]
)

# Representative coordinates for each CCAA capital / major city
REGIONS_COORDS = {
    'Andalucía': (37.3828, -5.9732), # Seville
    'Aragón': (41.6561, -0.8773), # Zaragoza
    'Cantabria': (43.4647, -3.8044), # Santander
    'Castilla la Mancha': (39.8628, -4.0273), # Toledo
    'Castilla y León': (41.6523, -4.7245), # Valladolid
    'Cataluña': (41.3888, 2.1590), # Barcelona
    'País Vasco': (42.8590, -2.6818), # Vitoria-Gasteiz
    'Principado de Asturias': (43.3614, -5.8593), # Oviedo
    'Comunidad de Ceuta': (35.8894, -5.3213), # Ceuta
    'Comunidad de Melilla': (35.2923, -2.9381), # Melilla
    'Comunidad de Madrid': (40.4165, -3.7026), # Madrid
    'Comunidad de Navarra': (42.8169, -1.6432), # Pamplona
    'Comunidad Valenciana': (39.4697, -0.3774), # Valencia
    'Extremadura': (38.9161, -6.3437), # Mérida
    'Galicia': (42.8805, -8.5457), # Santiago de Compostela
    'Islas Baleares': (39.5694, 2.6502), # Palma
    'Islas Canarias': (28.1235, -15.4363), # Las Palmas
    'La Rioja': (42.4627, -2.4450), # Logroño
    'Región de Murcia': (37.9870, -1.1300) # Murcia
}

import datetime
def main():
    start_date = "2015-01-01"
    # Use yesterday's date to avoid 400 Bad Request from Open-Meteo for future/current day
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = yesterday.strftime("%Y-%m-%d")
    
    raw_dir = os.path.join("data", "raw", "weather")
    proc_dir = os.path.join("data", "processed", "weather")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    for name, (lat, lon) in REGIONS_COORDS.items():
        logging.info(f"Downloading weather data for {name}...")
        safe_name = name.replace(" ", "_").replace("í", "i").replace("á", "a").replace("é", "e").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        csv_file = os.path.join(proc_dir, f"{safe_name}.csv")
        
        # Open-Meteo Historical Weather API
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
            f"&timezone=Europe/Madrid"
        )
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read().decode('utf-8'))
            
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            temps = hourly.get('temperature_2m', [])
            hums = hourly.get('relative_humidity_2m', [])
            winds = hourly.get('wind_speed_10m', [])
            rads = hourly.get('shortwave_radiation', [])
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["region", "datetime", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"])
                
                for i in range(len(times)):
                    writer.writerow([name, times[i], temps[i], hums[i], winds[i], rads[i]])
                    
            logging.info(f"Successfully saved weather data for {name}")
            
        except Exception as e:
            logging.error(f"Failed to fetch weather for {name}: {e}")
            
        time.sleep(2) # rate limit

if __name__ == "__main__":
    main()
