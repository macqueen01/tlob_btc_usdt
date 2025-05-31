import datetime
import pathlib
import zipfile

import requests
from tqdm import tqdm

BTC_USDT = "BTCUSDT"
SECOND_INTERVAL = "1s"
BINANCE_BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
DATA_DIR = pathlib.Path("./raw_data")

def build_dataset_url(
    name: str, 
    date: datetime.date,
    interval: str
) -> str:
    return f"{BINANCE_BASE_URL}/{name}/{interval}/{name}-{interval}-{date.strftime('%Y-%m-%d')}.zip"

def build_dataset_path(
    name: str, 
    interval: str,
    file_name: str
) -> pathlib.Path:
    return DATA_DIR / name / interval / file_name

def get_binance_data(
    date: datetime.date,
    name: str,
    dest: pathlib.Path
) -> list[dict]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = build_dataset_url(name, date, SECOND_INTERVAL)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest, "wb") as f, tqdm(
        desc=f"Downloading {name} {date} {SECOND_INTERVAL}",
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
            
if __name__ == "__main__":
    # date = datetime.date(2025, 5, 22)
    
    # dest = build_dataset_path(
    #     BTC_USDT,
    #     SECOND_INTERVAL,
    #     date.strftime("%Y-%m-%d") + ".zip"
    # )
    
    # get_binance_data(
    #     date=date,
    #     name=BTC_USDT,
    #     dest=dest
    # )
    
    # with zipfile.ZipFile(dest, "r") as zip_ref:
    #     zip_ref.extractall(dest.parent)
    
    # Read CSV file
    import pandas as pd
    
    csv_file_path = "raw_data/IDDI-5171075+SC-BINANCE_SPOT_BTC_USDC+S-BTCUSDC.csv"
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path, sep=';')
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Display data types
    print("\nData types:")
    print(df.dtypes)


