"""
Real-time Weather Data Integration with OpenWeather API
Fetches current and forecast rainfall for Delhi wards
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# OpenWeather API Configuration
OPENWEATHER_API_KEY = "76751d720ba3bf9800e42debcdd90948"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Delhi bounding box
DELHI_CENTER = {"lat": 28.6139, "lon": 77.2090}
DELHI_BBOX = {
    "lat_min": 28.4,
    "lat_max": 28.9,
    "lon_min": 76.8,
    "lon_max": 77.4
}


def fetch_current_weather_delhi() -> Dict:
    """Fetch current weather for Delhi."""
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {
        "lat": DELHI_CENTER["lat"],
        "lon": DELHI_CENTER["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract rainfall info
        rain_1h = data.get('rain', {}).get('1h', 0)  # mm in last hour
        rain_3h = data.get('rain', {}).get('3h', 0)  # mm in last 3 hours
        
        return {
            'timestamp': datetime.now(),
            'rain_1h': rain_1h,
            'rain_3h': rain_3h,
            'temperature': data.get('main', {}).get('temp', 0),
            'humidity': data.get('main', {}).get('humidity', 0),
            'description': data.get('weather', [{}])[0].get('description', ''),
            'pressure': data.get('main', {}).get('pressure', 0)
        }
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None


def fetch_forecast_delhi() -> List[Dict]:
    """Fetch 5-day forecast for Delhi (3-hour intervals)."""
    url = f"{OPENWEATHER_BASE_URL}/forecast"
    params = {
        "lat": DELHI_CENTER["lat"],
        "lon": DELHI_CENTER["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecasts = []
        for item in data.get('list', [])[:8]:  # Next 24 hours (8 x 3-hour periods)
            rain_3h = item.get('rain', {}).get('3h', 0)
            forecasts.append({
                'timestamp': datetime.fromtimestamp(item['dt']),
                'rain_3h': rain_3h,
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description']
            })
        
        return forecasts
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return []


def calculate_rainfall_features(current_weather: Dict, forecast: List[Dict]) -> Dict:
    """
    Calculate rainfall features from current weather and forecast.
    
    Returns features compatible with flood prediction model.
    """
    if not current_weather:
        # Fallback to defaults
        return {
            'rain_1h': 0,
            'rain_3h': 0,
            'rain_6h': 0,
            'rain_24h': 0,
            'rain_intensity': 0,
            'rain_forecast_3h': 0
        }
    
    # Current rainfall
    rain_1h = current_weather.get('rain_1h', 0)
    rain_3h = current_weather.get('rain_3h', 0)
    
    # Estimate 6h and 24h from current (in production, you'd store historical data)
    # For now, use conservative estimates
    rain_6h = rain_3h * 1.5  # Rough estimate
    rain_24h = rain_3h * 2.0  # Conservative estimate
    
    # Forecast for next 3 hours
    rain_forecast_3h = 0
    if forecast and len(forecast) > 0:
        # Sum next 3 hours of forecast
        rain_forecast_3h = forecast[0].get('rain_3h', 0)
    
    # Intensity
    rain_intensity = rain_1h if rain_1h > 0 else rain_3h / 3
    
    return {
        'rain_1h': rain_1h,
        'rain_3h': rain_3h,
        'rain_6h': rain_6h,
        'rain_24h': rain_24h,
        'rain_intensity': rain_intensity,
        'rain_forecast_3h': rain_forecast_3h
    }


def test_api_connection():
    """Test the OpenWeather API connection."""
    print("=" * 70)
    print("TESTING OPENWEATHER API CONNECTION")
    print("=" * 70)
    
    # Test current weather
    print("\n[1] Fetching current weather for Delhi...")
    current = fetch_current_weather_delhi()
    
    if current:
        print("✅ Successfully connected!")
        print(f"\n  Timestamp: {current['timestamp']}")
        print(f"  Rain (1h): {current['rain_1h']:.1f} mm")
        print(f"  Rain (3h): {current['rain_3h']:.1f} mm")
        print(f"  Temperature: {current['temperature']:.1f}°C")
        print(f"  Humidity: {current['humidity']}%")
        print(f"  Conditions: {current['description']}")
    else:
        print("❌ Failed to connect")
        return False
    
    # Test forecast
    print("\n[2] Fetching 24-hour forecast...")
    forecast = fetch_forecast_delhi()
    
    if forecast:
        print(f"✅ Got {len(forecast)} forecast periods")
        print("\n  Next 12 hours:")
        for f in forecast[:4]:
            print(f"    {f['timestamp'].strftime('%H:%M')}: {f['rain_3h']:.1f}mm, {f['temperature']:.1f}°C - {f['description']}")
    else:
        print("❌ Failed to get forecast")
        return False
    
    # Calculate features
    print("\n[3] Calculating rainfall features...")
    features = calculate_rainfall_features(current, forecast)
    print("✅ Features calculated:")
    for key, value in features.items():
        print(f"    {key}: {value:.2f} mm")
    
    print("\n" + "=" * 70)
    print("API CONNECTION SUCCESSFUL!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    test_api_connection()
