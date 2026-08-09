import fastf1
import requests

fastf1.Cache.enable_cache('cache')

# FastF1 - telemetry
session = fastf1.get_session(2026, 'Silverstone', 'R')
session.load()
fastest_lap = session.laps.pick_drivers('NOR').pick_fastest()
telemetry = fastest_lap.get_telemetry()

# OpenF1 - live data
r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': 1, 'session_key': 'latest'})
live_data = r.json()

print("FastF1 telemetry sample:")
print(telemetry[['Speed', 'Throttle', 'Brake', 'nGear', 'RPM']].head(5))

print("\nOpenF1 live sample:")
print(live_data[:5])