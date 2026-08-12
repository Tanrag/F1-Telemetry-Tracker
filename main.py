from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import requests
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

fastf1.Cache.enable_cache('cache')

_session_cache = {}

def get_cached_session():
    if 'session' not in _session_cache:
        session = fastf1.get_session(2026, 'Silverstone', 'R')
        session.load()
        _session_cache['session'] = session
    return _session_cache['session']

@app.get("/telemetry/{driver}")
def get_telemetry(driver: str):
    session = get_cached_session()
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    corner_distances = corners['Distance'].values
    corner_numbers = corners['Number'].values

    driver_number_to_name = {}
    for drv in session.drivers:
        info = session.get_driver(drv)
        driver_number_to_name[drv] = info['FullName']

    driver_laps = session.laps.pick_drivers(driver).iloc[10:15]
    result = {}

    for _, lap in driver_laps.iterrows():
        lap_number = int(lap['LapNumber'])

        telemetry = lap.get_telemetry()
        print(telemetry['DRS'].unique())
        telemetry = telemetry.add_driver_ahead()

        idx = np.searchsorted(corner_distances, telemetry['Distance'].values, side='right') - 1
        idx = np.clip(idx, 0, len(corner_numbers) - 1)
        telemetry['Turn'] = corner_numbers[idx]

        telemetry['SpeedMph'] = telemetry['Speed'] * 0.621371
        telemetry['DRSZone'] = telemetry['DRS'] == 8
        telemetry['DRSActive'] = telemetry['DRS'].isin([10, 12, 14])
        telemetry['DriverAheadName'] = telemetry['DriverAhead'].map(driver_number_to_name)

        speed_ms = telemetry['Speed'] / 3.6
        telemetry['GapToDriverAhead'] = telemetry['DistanceToDriverAhead'] / speed_ms.replace(0, np.nan)

        track_status_raw = str(lap['TrackStatus'])
        track_status_split = '-'.join(list(track_status_raw))

        result[lap_number] = {
            "lap_info": {
                "LapTime": str(lap['LapTime']),
                "Sector1Time": str(lap['Sector1Time']),
                "Sector2Time": str(lap['Sector2Time']),
                "Sector3Time": str(lap['Sector3Time']),
                "Compound": lap['Compound'],
                "TyreLife": lap['TyreLife'],
                "Stint": lap['Stint'],
                "Position": lap['Position'],
                "TrackStatus": track_status_split,
                "IsAccurate": bool(lap['IsAccurate'])
            },
            "telemetry": json.loads(telemetry[[
    'SpeedMph', 'Throttle', 'Brake', 'nGear', 'RPM', 'DRS', 'DRSZone', 'DRSActive',
    'Distance', 'Turn', 'DriverAheadName', 'GapToDriverAhead'
]].to_json())
        }

    return result

@app.get("/live/{driver_number}")
def get_live_data(driver_number: int):
    r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': driver_number, 'session_key': 'latest'})
    return r.json()[:20]