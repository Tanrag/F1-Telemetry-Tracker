from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

fastf1.Cache.enable_cache('cache')

@app.get("/telemetry/{driver}")
def get_telemetry(driver: str):
    session = fastf1.get_session(2026, 'Silverstone', 'R')
    session.load()
    lap = session.laps.pick_drivers(driver).pick_fastest()
    telemetry = lap.get_telemetry()
    return telemetry[['Speed', 'Throttle', 'Brake', 'nGear', 'RPM']].head(20).to_dict()

@app.get("/live/{driver_number}")
def get_live_data(driver_number: int):
    r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': driver_number, 'session_key': 'latest'})
    return r.json()[:20]