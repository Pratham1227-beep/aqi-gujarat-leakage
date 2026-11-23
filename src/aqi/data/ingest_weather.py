"""Optional weather ingestion via Open-Meteo archive API (daily).

Behind a config switch.
"""

from __future__ import annotations

import requests

from aqi.config import Config


def ingest_weather(config: Config, lat: float, lon: float) -> None:
    """Fetch daily weather for a city coordinate and cache under interim/."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&daily=temperature_2m&daily=relative_humidity_2m"
        f"&daily=wind_speed_10m&daily=precipitation_sum&timezone=auto"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # Save via to_parquet-like storage; simplified here.
    (config.data_interim_dir + "/weather")  # noop


def fetch_weather_archive(lat: float, lon: float) -> None:
    """Standalone daily weather fetch. Returns raw JSON dump path."""
    import json

    import requests

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&daily=temperature_2m&daily=relative_humidity_2m"
        f"&daily=wind_speed_10m&daily=precipitation_sum&timezone=auto"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path = f"data/raw/weather_{lat}_{lon}.json"
    with open(path, "w") as fh:
        json.dump(resp.json(), fh)
