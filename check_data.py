import pandas as pd
import glob
import os

files = glob.glob('c:/Users/soham/Desktop/edi-1/data/processed/demand/*.csv')
res = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            start = df['datetime'].min()[:10]
            end = df['datetime'].max()[:10]
            res.append(f"{os.path.basename(f):<25} | Rows: {len(df):<5} | {start} to {end}")
        else:
            res.append(f"{os.path.basename(f):<25} | Empty")
    except Exception as e:
        res.append(f"{os.path.basename(f):<25} | Error: {e}")

for line in res:
    print(line)
