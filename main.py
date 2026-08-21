from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import requests
import numpy as np
import json
import pandas as pd
import bisect
from contextlib import asynccontextmanager
import os
from concurrent.futures import ThreadPoolExecutor
import pickle

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

CACHE_FILE = os.path.join(CACHE_DIR, "lap_frame_cache.pkl")


CACHE_SCHEMA_VERSION = 2

def _load_lap_frame_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and data.get("_version") == CACHE_SCHEMA_VERSION:
            return data["frames"]
        print("Lap frame cache schema changed — discarding stale disk cache and recomputing.")
    return {}


def _save_lap_frame_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"_version": CACHE_SCHEMA_VERSION, "frames": _lap_frame_cache}, f)


def _save_lap_frame_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(_lap_frame_cache, f)


_session_cache = {}
_replay_cache = {}
_lap_frame_cache = _load_lap_frame_cache()


def get_cached_session():
    if 'session' not in _session_cache:
        session = fastf1.get_session(2026, 10, 'R')
        session.load()
        if len(session.drivers) == 0:
            raise HTTPException(
                status_code=503,
                detail="FastF1 loaded this session with 0 drivers — the event/round is likely wrong or data isn't available yet."
            )
        _session_cache['session'] = session
    return _session_cache['session']


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_cached_session()
    yield
    _save_lap_frame_cache()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_cached_session():
    if 'session' not in _session_cache:
        session = fastf1.get_session(2026, 10, 'R')
        session.load()
        if len(session.drivers) == 0:
            raise HTTPException(
                status_code=503,
                detail="FastF1 loaded this session with 0 drivers — the event/round is likely wrong or data isn't available yet."
            )
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
            "telemetry": json.loads(telemetry[['SpeedMph', 'Throttle', 'Brake', 'nGear', 'RPM', 'DRS', 'DRSZone', 'DRSActive', 'Distance', 'Turn', 'DriverAheadName', 'GapToDriverAhead']].to_json())
        }

    return result


@app.get("/live/{driver_number}")
def get_live_data(driver_number: int):
    r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': driver_number, 'session_key': 'latest'})
    return r.json()[:20]


@app.get("/track-outline")
def get_track_outline():
    session = get_cached_session()
    fastest_lap = session.laps.pick_fastest()
    tel = fastest_lap.get_telemetry()

    xs = tel['X'].values
    ys = tel['Y'].values

    bounds = {
        "min_x": float(xs.min()), "max_x": float(xs.max()),
        "min_y": float(ys.min()), "max_y": float(ys.max()),
    }
    _session_cache['track_bounds'] = bounds

    points = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
    return {"points": points, "bounds": bounds}


def lap_at_time(frames, t):
    if not frames:
        return None
    idx = bisect.bisect_right([f['t'] for f in frames], t) - 1
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]['lap']


def _split_bulk_telemetry_into_lap_frames(drv, missing_slice, driver_number_to_name):
    lap_numbers = sorted(int(n) for n in missing_slice['LapNumber'])
    print(f"  bulk-computing driver {drv} laps {lap_numbers[0]}-{lap_numbers[-1]} "
          f"({len(lap_numbers)} laps in a single telemetry call)")

    try:
        telemetry = missing_slice.get_telemetry()
        if telemetry.empty:
            raise ValueError("bulk telemetry returned empty")
    except Exception as e:
        print(f"  WARNING: bulk fetch failed for driver {drv} laps {lap_numbers}: {e!r}")
        print(f"  Falling back to per-lap fetch for driver {drv}...")
        _fetch_laps_individually(drv, missing_slice, driver_number_to_name)
        return

    telemetry = telemetry.add_driver_ahead()
    speed_ms = telemetry['Speed'] / 3.6
    telemetry['GapToDriverAhead'] = telemetry['DistanceToDriverAhead'] / speed_ms.replace(0, np.nan)
    telemetry['DriverAheadName'] = telemetry['DriverAhead'].map(driver_number_to_name)

    lap_starts = missing_slice[['LapNumber', 'LapStartTime', 'Position', 'Compound', 'PitInTime', 'PitOutTime']].sort_values('LapStartTime').copy()
    lap_starts['Position'] = lap_starts['Position'].ffill().bfill()
    telemetry = telemetry.sort_values('SessionTime')
    telemetry = pd.merge_asof(
        telemetry, lap_starts,
        left_on='SessionTime', right_on='LapStartTime',
        direction='backward'
    )

    telemetry['t'] = telemetry['SessionTime'].dt.total_seconds()
    telemetry['gap_r'] = telemetry['GapToDriverAhead'].round(2)

    by_lap = {lap_number: [] for lap_number in lap_numbers}
    for row in telemetry.to_dict('records'):
        lap_number = row.get('LapNumber')
        if lap_number is None or (isinstance(lap_number, float) and np.isnan(lap_number)):
            continue
        lap_number = int(lap_number)

        pos = row['Position']
        gap = row['gap_r']
        ahead_num = row['DriverAhead'] if row['DriverAhead'] else None

        row_time = row['SessionTime']
        pit_in = row.get('PitInTime')
        pit_out = row.get('PitOutTime')

        in_pit = False
        if pit_in is not None and not pd.isna(pit_in) and row_time >= pit_in:
            in_pit = True
        if pit_out is not None and not pd.isna(pit_out) and row_time <= pit_out:
            in_pit = True

        by_lap.setdefault(lap_number, []).append({
            "t": row['t'],
            "x": float(row['X']),
            "y": float(row['Y']),
            "lap": lap_number,
            "position": None if pd.isna(pos) else int(pos),
            "compound": row['Compound'],
            "gap": None if pd.isna(gap) else float(gap),
            "driverAhead": row['DriverAheadName'] if isinstance(row['DriverAheadName'], str) else None,
            "driverAheadNum": ahead_num,
            "inPit": bool(in_pit),
        })

    for lap_number, frames in by_lap.items():
        _lap_frame_cache[(drv, lap_number)] = frames


def _fetch_laps_individually(drv, missing_slice, driver_number_to_name):
    for _, lap in missing_slice.iterrows():
        lap_number = int(lap['LapNumber'])
        try:
            telemetry = lap.get_telemetry()
            if telemetry.empty:
                raise ValueError("empty telemetry")
        except Exception as e:
            print(f"    driver {drv} lap {lap_number}: no usable telemetry ({e!r}) — leaving empty")
            _lap_frame_cache[(drv, lap_number)] = []
            continue

        telemetry = telemetry.add_driver_ahead()
        speed_ms = telemetry['Speed'] / 3.6
        telemetry['GapToDriverAhead'] = telemetry['DistanceToDriverAhead'] / speed_ms.replace(0, np.nan)
        telemetry['DriverAheadName'] = telemetry['DriverAhead'].map(driver_number_to_name)

        position = lap['Position']
        compound = lap['Compound']
        pit_in = lap.get('PitInTime')
        pit_out = lap.get('PitOutTime')

        frames = []
        for _, row in telemetry.iterrows():
            ahead_num = row['DriverAhead'] if row['DriverAhead'] else None

            row_time = row['SessionTime']
            in_pit = False
            if pit_in is not None and not pd.isna(pit_in) and row_time >= pit_in:
                in_pit = True
            if pit_out is not None and not pd.isna(pit_out) and row_time <= pit_out:
                in_pit = True

            frames.append({
                "t": row_time.total_seconds(),
                "x": float(row['X']),
                "y": float(row['Y']),
                "lap": lap_number,
                "position": None if pd.isna(position) else int(position),
                "compound": compound,
                "gap": None if pd.isna(row['GapToDriverAhead']) else round(float(row['GapToDriverAhead']), 2),
                "driverAhead": row['DriverAheadName'] if isinstance(row['DriverAheadName'], str) else None,
                "driverAheadNum": ahead_num,
                "inPit": bool(in_pit),
            })
        _lap_frame_cache[(drv, lap_number)] = frames
        print(f"    driver {drv} lap {lap_number}: {len(frames)} frames recovered individually")


def compute_driver_frames(drv, drv_laps, driver_number_to_name):
    missing_slice = drv_laps[[
        (drv, int(lap['LapNumber'])) not in _lap_frame_cache
        for _, lap in drv_laps.iterrows()
    ]]

    if not missing_slice.empty:
        _split_bulk_telemetry_into_lap_frames(drv, missing_slice, driver_number_to_name)

    frames = []
    for _, lap in drv_laps.iterrows():
        frames.extend(_lap_frame_cache.get((drv, int(lap['LapNumber'])), []))
    return frames


@app.get("/replay")
def get_replay(start_lap: int, end_lap: int):
    if end_lap < start_lap:
        start_lap, end_lap = end_lap, start_lap

    cache_key = (start_lap, end_lap)
    if cache_key in _replay_cache:
        print(f"Serving cached replay for laps {start_lap}-{end_lap}")
        return _replay_cache[cache_key]

    session = get_cached_session()
    laps = session.laps

    driver_number_to_name = {
        drv: session.get_driver(drv)['FullName'] for drv in session.drivers
    }

    driver_info = {}
    for drv in session.drivers:
        info = session.get_driver(drv)
        all_drv_laps = laps.pick_drivers(drv)
        last_lap = int(all_drv_laps['LapNumber'].max()) if not all_drv_laps.empty else 0
        driver_info[drv] = {"name": info['FullName'], "team": info['TeamName'], "lastLap": last_lap}

    def process_driver(drv):
        all_drv_laps = laps.pick_drivers(drv)
        drv_laps = all_drv_laps[(all_drv_laps['LapNumber'] >= start_lap) & (all_drv_laps['LapNumber'] <= end_lap)]
        frames = compute_driver_frames(drv, drv_laps, driver_number_to_name)
        frames.sort(key=lambda f: f['t'])
        return drv, frames

    frames_by_driver = {}
    with ThreadPoolExecutor(max_workers=min(22, len(session.drivers))) as pool:
        for drv, frames in pool.map(process_driver, session.drivers):
            frames_by_driver[drv] = frames

    for drv, frames in frames_by_driver.items():
        new_frames = []
        for f in frames:
            f = dict(f) 
            ahead_num = f.pop("driverAheadNum")
            lapped = False
            if ahead_num and ahead_num in frames_by_driver:
                ahead_lap = lap_at_time(frames_by_driver[ahead_num], f['t'])
                if ahead_lap is not None and ahead_lap != f['lap']:
                    lapped = True
            f["lapped"] = lapped
            if lapped:
                f["gap"] = None
            new_frames.append(f)
        frames_by_driver[drv] = new_frames

    result = {"driver_info": driver_info, "frames": frames_by_driver}
    _replay_cache[cache_key] = result
    print(f"Finished and cached replay for laps {start_lap}-{end_lap}")
    return result

