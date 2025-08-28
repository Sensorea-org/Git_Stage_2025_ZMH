import requests
import pandas as pd
from datetime import datetime


def get_temp(date_datetime,lat =50.8503, lon = 4.3517):
    date = str(date_datetime)
    date = date.split(" ")
    date = date[0]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m"
    )

    response = requests.get(url)
    data = response.json()
    temps = data['hourly']['temperature_2m']
    timestamps = data['hourly']['time']
    df = pd.DataFrame({'time': timestamps, 'temperature': temps})
    df['time'] = pd.to_datetime(df['time'])

    return df['temperature']

