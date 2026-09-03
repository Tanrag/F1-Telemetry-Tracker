from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import requests
import numpy as np
import json
import pandas as pd
import bisect
from contextlib import asynccontextmanager
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

CACHE_FILE = os.path.join(CACHE_DIR, "lap_frame_cache.pkl")

CACHE_SCHEMA_VERSION = 7

TARGET_TELEMETRY_TEAMS = {"McLaren", "Red Bull Racing", "Ferrari", "Mercedes"}

TELEMETRY_CHANNEL_COLUMNS = [
    'SpeedMph', 'Throttle', 'Brake', 'nGear', 'RPM', 'DRS', 'DRSZone',
    'DRSActive', 'Distance', 'DriverAheadName', 'GapToDriverAhead'
]


def _load_lap_frame_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Lap frame cache file is corrupt or unreadable ({e!r}) — discarding and starting fresh.")
            try:
                os.remove(CACHE_FILE)
            except OSError:
                pass
            return {}, {}
        if isinstance(data, dict) and data.get("_version") == CACHE_SCHEMA_VERSION:
            return data["frames"], data.get("telemetry", {})
        print("Lap frame cache schema changed — discarding stale disk cache and recomputing.")
    return {}, {}


def _save_lap_frame_cache():
    tmp_path = CACHE_FILE + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({
            "_version": CACHE_SCHEMA_VERSION,
            "frames": _lap_frame_cache,
            "telemetry": _telemetry_cache,
        }, f)
    os.replace(tmp_path, CACHE_FILE)


_session_cache = {}
_replay_cache = {}
_lap_frame_cache, _telemetry_cache = _load_lap_frame_cache()
_fastest_lap_cache = {}
_track_status_cache = {}

_cache_lock = threading.Lock()
_fastest_lap_cache_lock = threading.Lock()
_track_status_cache_lock = threading.Lock()

TRACK_STATUS_LABELS = {
    '1': ('clear', 'Track Clear'),
    '2': ('yellow', 'Yellow Flag'),
    '4': ('safety_car', 'Safety Car Deployed'),
    '5': ('red', 'Red Flag'),
    '6': ('vsc', 'Virtual Safety Car Deployed'),
    '7': ('vsc_ending', 'VSC Ending'),
}

_warmed_rounds = set()
_warmed_rounds_lock = threading.Lock()

# --- OpenF1 session_key lookup + team radio caches -------------------------
_openf1_session_key_cache = {}
_openf1_session_key_lock = threading.Lock()

_radio_cache = {}
_radio_cache_lock = threading.Lock()


def get_cached_session(round_number: int):
    if round_number not in _session_cache:
        session = fastf1.get_session(2026, round_number, 'R')
        session.load()
        if len(session.drivers) == 0:
            raise HTTPException(
                status_code=503,
                detail=f"FastF1 loaded round {round_number} with 0 drivers — "
                       f"the event is likely wrong or data isn't available yet."
            )
        _session_cache[round_number] = session

    return _session_cache[round_number]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _save_lap_frame_cache()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_events_cache = None
_events_cache_lock = threading.Lock()


@app.get("/events")
def get_events():
    global _events_cache
    with _events_cache_lock:
        if _events_cache is not None:
            return _events_cache

    try:
        schedule = fastf1.get_event_schedule(2026)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not fetch event schedule: {e!r}")

    schedule = schedule[schedule['RoundNumber'] > 0]

    now = pd.Timestamp.now()
    events = []
    for _, row in schedule.iterrows():
        event_date = row.get('EventDate')
        completed = False
        date_str = None
        if event_date is not None and not pd.isna(event_date):
            date_str = str(event_date.date())
            try:
                completed = bool(event_date.tz_localize(None) < now) if event_date.tzinfo else bool(event_date < now)
            except Exception:
                completed = bool(event_date < now)

        events.append({
            "round": int(row['RoundNumber']),
            "name": row['EventName'],
            "location": row['Location'],
            "date": date_str,
            "completed": completed,
        })

    result = {"events": events}
    with _events_cache_lock:
        _events_cache = result
    return result


def _lap_info_dict(lap):
    track_status_raw = str(lap['TrackStatus'])
    track_status_split = '-'.join(list(track_status_raw))
    return {
        "LapTime": str(lap['LapTime']),
        "Sector1Time": str(lap['Sector1Time']),
        "Sector2Time": str(lap['Sector2Time']),
        "Sector3Time": str(lap['Sector3Time']),
        "Compound": lap['Compound'],
        "TyreLife": lap['TyreLife'],
        "Stint": lap['Stint'],
        "Position": lap['Position'],
        "TrackStatus": track_status_split,
        "IsAccurate": bool(lap['IsAccurate']),
    }


def _store_full_telemetry_channels(round_number, drv, lap_number, lap_telemetry, lap_row):
    t = lap_telemetry.copy()
    t['SpeedMph'] = t['Speed'] * 0.621371
    t['DRSZone'] = t['DRS'] == 8
    t['DRSActive'] = t['DRS'].isin([10, 12, 14])
    cols = json.loads(t[TELEMETRY_CHANNEL_COLUMNS].to_json())
    _telemetry_cache[(round_number, drv, lap_number)] = {
        "lap_info": _lap_info_dict(lap_row),
        "telemetry_cols": cols,
    }


def _turn_for_distances(session, distance_values):
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners
    corner_distances = corners['Distance'].values
    corner_numbers = corners['Number'].values
    idx = np.searchsorted(corner_distances, distance_values, side='right') - 1
    idx = np.clip(idx, 0, len(corner_numbers) - 1)
    return corner_numbers[idx]


@app.get("/telemetry/{driver}")
def get_telemetry(driver: str, round_number: int = Query(..., alias="round")):
    session = get_cached_session(round_number)

    driver_number_to_name = {}
    for drv in session.drivers:
        info = session.get_driver(drv)
        driver_number_to_name[drv] = info['FullName']

    driver_laps = session.laps.pick_drivers(driver).iloc[10:15]
    result = {}

    for _, lap in driver_laps.iterrows():
        lap_number = int(lap['LapNumber'])

        with _cache_lock:
            cached = _telemetry_cache.get((round_number, driver, lap_number))

        if cached is not None:
            distances = list(cached["telemetry_cols"]["Distance"].values())
            turn_values = _turn_for_distances(session, np.array(distances, dtype=float))
            turn_by_index = dict(zip(cached["telemetry_cols"]["Distance"].keys(), [int(v) for v in turn_values]))
            telemetry_out = dict(cached["telemetry_cols"])
            telemetry_out["Turn"] = turn_by_index
            result[lap_number] = {"lap_info": cached["lap_info"], "telemetry": telemetry_out}
            continue

        telemetry = lap.get_telemetry()
        telemetry = telemetry.add_driver_ahead()

        telemetry['Turn'] = _turn_for_distances(session, telemetry['Distance'].values)

        telemetry['SpeedMph'] = telemetry['Speed'] * 0.621371
        telemetry['DRSZone'] = telemetry['DRS'] == 8
        telemetry['DRSActive'] = telemetry['DRS'].isin([10, 12, 14])
        telemetry['DriverAheadName'] = telemetry['DriverAhead'].map(driver_number_to_name)

        speed_ms = telemetry['Speed'] / 3.6
        telemetry['GapToDriverAhead'] = telemetry['DistanceToDriverAhead'] / speed_ms.replace(0, np.nan)
        telemetry['SpeedMph'] = telemetry['Speed'] * 0.621371

        result[lap_number] = {
            "lap_info": _lap_info_dict(lap),
            "telemetry": json.loads(telemetry[['SpeedMph', 'Throttle', 'Brake', 'nGear', 'RPM', 'DRS', 'DRSZone', 'DRSActive', 'Distance', 'Turn', 'DriverAheadName', 'GapToDriverAhead']].to_json())
        }

    return result


@app.get("/live/{driver_number}")
def get_live_data(driver_number: int):
    r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': driver_number, 'session_key': 'latest'})
    return r.json()[:20]


@app.get("/track-outline")
def get_track_outline(round_number: int = Query(..., alias="round")):
    session = get_cached_session(round_number)
    laps = session.laps

    accurate = laps[laps['IsAccurate'] == True]
    candidates = accurate if not accurate.empty else laps
    fastest_lap = candidates.pick_fastest()
    tel = fastest_lap.get_telemetry()

    if len(tel) < 100 and not accurate.empty:
        best_lap, best_len = None, 0
        for _, lap in accurate.iterrows():
            t = lap.get_telemetry()
            if len(t) > best_len:
                best_len, best_lap = len(t), t
        if best_lap is not None:
            tel = best_lap

    xs, ys, distances = tel['X'].values, tel['Y'].values, tel['Distance'].values

    bounds = {
        "min_x": float(xs.min()), "max_x": float(xs.max()),
        "min_y": float(ys.min()), "max_y": float(ys.max()),
    }
    points = [{"x": float(x), "y": float(y), "distance": float(d)}
              for x, y, d in zip(xs, ys, distances)]

    return {
        "points": points,
        "bounds": bounds,
        "straightZones": _compute_straight_zones(session, float(distances.max())),
        "pitLane": _compute_pit_lane_path(session),
    }


def lap_at_time(frames, t):
    if not frames:
        return None
    idx = bisect.bisect_right([f['t'] for f in frames], t) - 1
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]['lap']


def _format_laptime(td):
    """Timedelta -> 'm:ss.mmm'. Used for the fastest-lap toast, which needs
    something readable rather than the raw '0 days 00:01:12...' repr."""
    if td is None or pd.isna(td):
        return None
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def _compute_fastest_lap_events(round_number: int):
    with _fastest_lap_cache_lock:
        cached = _fastest_lap_cache.get(round_number)
    if cached is not None:
        return cached

    session = get_cached_session(round_number)
    laps = session.laps

    driver_number_to_team = {
        drv: session.get_driver(drv)['TeamName'] for drv in session.drivers
    }

    valid = laps[laps['LapTime'].notna() & laps['Time'].notna()].copy()
    valid = valid.sort_values('Time')

    events = []
    best = None
    for _, lap in valid.iterrows():
        lap_time = lap['LapTime']
        if best is None or lap_time < best:
            best = lap_time
            events.append({
                "t": float(lap['Time'].total_seconds()),
                "drv": lap['DriverNumber'],
                "lap": int(lap['LapNumber']),
                "lapTime": _format_laptime(lap_time),
                "team": driver_number_to_team.get(lap['DriverNumber']),
            })

    with _fastest_lap_cache_lock:
        _fastest_lap_cache[round_number] = events
    return events


def _compute_track_status_events(round_number: int):
    with _track_status_cache_lock:
        cached = _track_status_cache.get(round_number)
    if cached is not None:
        return cached

    session = get_cached_session(round_number)
    ts = session.track_status

    events = []
    if ts is not None and not ts.empty:
        ts = ts.sort_values('Time')
        for _, row in ts.iterrows():
            t = row['Time']
            if t is None or pd.isna(t):
                continue
            code = str(row['Status'])
            kind, label = TRACK_STATUS_LABELS.get(code, ('clear', f'Status {code}'))
            events.append({
                "t": float(t.total_seconds()),
                "code": code,
                "kind": kind,
                "label": label,
            })

    with _track_status_cache_lock:
        _track_status_cache[round_number] = events
    return events


def _split_bulk_telemetry_into_lap_frames(round_number, drv, missing_slice, driver_number_to_name, driver_number_to_team=None):
    lap_numbers = sorted(int(n) for n in missing_slice['LapNumber'])
    print(f"  bulk-computing round {round_number} driver {drv} laps {lap_numbers[0]}-{lap_numbers[-1]} "
          f"({len(lap_numbers)} laps in a single telemetry call)")

    try:
        telemetry = missing_slice.get_telemetry()
        if telemetry.empty:
            raise ValueError("bulk telemetry returned empty")
    except Exception as e:
        print(f"  WARNING: bulk fetch failed for round {round_number} driver {drv} laps {lap_numbers}: {e!r}")
        print(f"  Falling back to per-lap fetch for driver {drv}...")
        _fetch_laps_individually(round_number, drv, missing_slice, driver_number_to_name, driver_number_to_team)
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

    team = (driver_number_to_team or {}).get(drv)
    if team in TARGET_TELEMETRY_TEAMS:
        lap_info_by_number = {int(lap['LapNumber']): lap for _, lap in missing_slice.iterrows()}
        for lap_number, group in telemetry.groupby('LapNumber'):
            lap_number = int(lap_number)
            lap_row = lap_info_by_number.get(lap_number)
            if lap_row is None or group.empty:
                continue
            _store_full_telemetry_channels(round_number, drv, lap_number, group, lap_row)

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
            "throttle": None if pd.isna(row.get('Throttle')) else float(row['Throttle']),
            "brake": None if pd.isna(row.get('Brake')) else bool(row['Brake']),
            "speed": None if pd.isna(row.get('SpeedMph')) else float(row['SpeedMph']),
        })

    for lap_number, frames in by_lap.items():
        _lap_frame_cache[(round_number, drv, lap_number)] = frames


def _fetch_laps_individually(round_number, drv, missing_slice, driver_number_to_name, driver_number_to_team=None):
    team = (driver_number_to_team or {}).get(drv)

    for _, lap in missing_slice.iterrows():
        lap_number = int(lap['LapNumber'])
        try:
            telemetry = lap.get_telemetry()
            if telemetry.empty:
                raise ValueError("empty telemetry")
        except Exception as e:
            print(f"    round {round_number} driver {drv} lap {lap_number}: no usable telemetry ({e!r}) — leaving empty")
            _lap_frame_cache[(round_number, drv, lap_number)] = []
            continue

        telemetry = telemetry.add_driver_ahead()
        speed_ms = telemetry['Speed'] / 3.6
        telemetry['GapToDriverAhead'] = telemetry['DistanceToDriverAhead'] / speed_ms.replace(0, np.nan)
        telemetry['DriverAheadName'] = telemetry['DriverAhead'].map(driver_number_to_name)

        if team in TARGET_TELEMETRY_TEAMS:
            _store_full_telemetry_channels(round_number, drv, lap_number, telemetry, lap)

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
                "throttle": None if pd.isna(row.get('Throttle')) else float(row['Throttle']),
                "brake": None if pd.isna(row.get('Brake')) else bool(row['Brake']),
                "speed": None if pd.isna(row.get('SpeedMph')) else float(row['SpeedMph']),
            })
        _lap_frame_cache[(round_number, drv, lap_number)] = frames
        print(f"    round {round_number} driver {drv} lap {lap_number}: {len(frames)} frames recovered individually")


def compute_driver_frames(round_number, drv, drv_laps, driver_number_to_name, driver_number_to_team=None):
    with _cache_lock:
        missing_slice = drv_laps[[
            (round_number, drv, int(lap['LapNumber'])) not in _lap_frame_cache
            for _, lap in drv_laps.iterrows()
        ]]

    if not missing_slice.empty:
        _split_bulk_telemetry_into_lap_frames(round_number, drv, missing_slice, driver_number_to_name, driver_number_to_team)

    with _cache_lock:
        frames = []
        for _, lap in drv_laps.iterrows():
            frames.extend(_lap_frame_cache.get((round_number, drv, int(lap['LapNumber'])), []))
    return frames


def _compute_replay(round_number: int, start_lap: int, end_lap: int):
    cache_key = (round_number, start_lap, end_lap)
    with _cache_lock:
        cached = _replay_cache.get(cache_key)
    if cached is not None:
        print(f"Serving cached replay for round {round_number} laps {start_lap}-{end_lap}")
        return cached

    session = get_cached_session(round_number)
    laps = session.laps

    driver_number_to_name = {
        drv: session.get_driver(drv)['FullName'] for drv in session.drivers
    }

    driver_info = {}
    for drv in session.drivers:
        info = session.get_driver(drv)
        all_drv_laps = laps.pick_drivers(drv)
        last_lap = int(all_drv_laps['LapNumber'].max()) if not all_drv_laps.empty else 0

        grid_position = info.get('GridPosition') if hasattr(info, 'get') else None
        if grid_position is not None and not pd.isna(grid_position):
            grid_position = int(grid_position)
        else:
            grid_position = None

        driver_info[drv] = {
            "name": info['FullName'],
            "team": info['TeamName'],
            "lastLap": last_lap,
            "gridPosition": grid_position,
        }

    driver_number_to_team = {drv: driver_info[drv]["team"] for drv in session.drivers}

    def process_driver(drv):
        all_drv_laps = laps.pick_drivers(drv)
        drv_laps = all_drv_laps[(all_drv_laps['LapNumber'] >= start_lap) & (all_drv_laps['LapNumber'] <= end_lap)]
        frames = compute_driver_frames(round_number, drv, drv_laps, driver_number_to_name, driver_number_to_team)
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
    with _cache_lock:
        _replay_cache[cache_key] = result
    print(f"Finished and cached replay for round {round_number} laps {start_lap}-{end_lap}")
    return result


def _prewarm_full_race(round_number: int):
    with _warmed_rounds_lock:
        if round_number in _warmed_rounds:
            return
        _warmed_rounds.add(round_number)

    def run():
        try:
            session = get_cached_session(round_number)
            total_laps = int(session.laps['LapNumber'].max())
            print(f"Prewarming round {round_number} in background: laps 1-{total_laps}")
            _compute_replay(round_number, 1, total_laps)
            _compute_fastest_lap_events(round_number)
            _compute_track_status_events(round_number)
            print(f"Round {round_number} background prewarm complete.")
        except Exception as e:
            print(f"WARNING: background prewarm for round {round_number} failed: {e!r}")
            with _warmed_rounds_lock:
                _warmed_rounds.discard(round_number)

    threading.Thread(target=run, daemon=True).start()


@app.get("/replay")
def get_replay(round_number: int = Query(..., alias="round"), start_lap: int = Query(...), end_lap: int = Query(...)):
    if end_lap < start_lap:
        start_lap, end_lap = end_lap, start_lap

    result = _compute_replay(round_number, start_lap, end_lap)

    fl_events = _compute_fastest_lap_events(round_number)
    fastest_laps = [e for e in fl_events if start_lap <= e["lap"] <= end_lap]
    session = get_cached_session(round_number)
    laps_window = session.laps[
        (session.laps['LapNumber'] >= start_lap) & (session.laps['LapNumber'] <= end_lap)
    ]
    window_start = laps_window['LapStartTime'].min() if not laps_window.empty else None
    window_end = laps_window['Time'].max() if not laps_window.empty else None

    all_ts_events = _compute_track_status_events(round_number)
    track_status_events = []
    initial_track_status = None
    if window_start is not None and window_end is not None and not pd.isna(window_start) and not pd.isna(window_end):
        w_start = float(window_start.total_seconds())
        w_end = float(window_end.total_seconds())
        track_status_events = [e for e in all_ts_events if w_start <= e["t"] <= w_end]
        before_window = [e for e in all_ts_events if e["t"] <= w_start]
        if before_window:
            initial_track_status = before_window[-1]

    _prewarm_full_race(round_number)

    return {
        **result,
        "fastestLaps": fastest_laps,
        "trackStatusEvents": track_status_events,
        "initialTrackStatus": initial_track_status,
    }


# --- Team radio --------------------------------------------------------
def _get_openf1_session_key(round_number: int):
    """OpenF1 keys sessions by session_key, not our round_number, so we
    resolve it once per round (via year + country + session name) and
    cache the result — including a cached None when nothing matches."""
    with _openf1_session_key_lock:
        cached = _openf1_session_key_cache.get(round_number)
        if round_number in _openf1_session_key_cache:
            return cached

    session = get_cached_session(round_number)
    country = session.event.get('Country')

    try:
        r = requests.get('https://api.openf1.org/v1/sessions', params={
            'year': 2026,
            'session_name': 'Race',
            'country_name': country,
        })
        results = r.json()
    except Exception as e:
        print(f"WARNING: OpenF1 session lookup failed for round {round_number}: {e!r}")
        results = []

    session_key = results[0]['session_key'] if results else None
    if session_key is None:
        print(f"No OpenF1 session_key found for round {round_number} ({country}) — "
              f"radio will be unavailable for this round.")

    with _openf1_session_key_lock:
        _openf1_session_key_cache[round_number] = session_key
    return session_key


@app.get("/radio/{driver_number}")
def get_team_radio(driver_number: int, round_number: int = Query(..., alias="round")):
    cache_key = (round_number, driver_number)
    with _radio_cache_lock:
        cached = _radio_cache.get(cache_key)
    if cached is not None:
        return {"messages": cached}

    session_key = _get_openf1_session_key(round_number)
    if session_key is None:
        with _radio_cache_lock:
            _radio_cache[cache_key] = []
        return {"messages": []}

    try:
        r = requests.get('https://api.openf1.org/v1/team_radio', params={
            'session_key': session_key,
            'driver_number': driver_number,
        })
        raw = r.json()
    except Exception as e:
        print(f"WARNING: OpenF1 team_radio fetch failed for round {round_number} driver {driver_number}: {e!r}")
        raw = []

    session = get_cached_session(round_number)
    t0 = session.t0_date
    if t0.tzinfo is None:
        t0 = t0.tz_localize('UTC')

    messages = []
    for msg in raw:
        try:
            msg_time = pd.Timestamp(msg['date'])
            if msg_time.tzinfo is None:
                msg_time = msg_time.tz_localize('UTC')
            t_offset = (msg_time - t0).total_seconds()
        except Exception:
            t_offset = None
        messages.append({
            "t": t_offset,
            "date": msg['date'],
            "recordingUrl": msg['recording_url'],
        })
    messages.sort(key=lambda m: m['t'] if m['t'] is not None else 0)

    with _radio_cache_lock:
        _radio_cache[cache_key] = messages
    return {"messages": messages}

def _compute_straight_zones(session, lap_length, min_length=250):
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners.sort_values('Distance')
    corner_distances = corners['Distance'].values.tolist()
    if not corner_distances:
        return []

    segments = [(corner_distances[i], corner_distances[i + 1])
                for i in range(len(corner_distances) - 1)]
    segments.append((corner_distances[-1], corner_distances[0] + lap_length))

    ranges = []
    for start, end in segments:
        if (end - start) < min_length:
            continue
        if end <= lap_length:
            ranges.append({"start": float(start), "end": float(end)})
        else:
            ranges.append({"start": float(start), "end": float(lap_length)})
            ranges.append({"start": 0.0, "end": float(end - lap_length)})
    return ranges

def _compute_pit_lane_path(session):
    laps = session.laps

    entry_points = []
    for _, lap in laps[laps['PitInTime'].notna()].iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        pit_in = lap['PitInTime']
        if tel.empty or pd.isna(pit_in):
            continue
        seg = tel[tel['SessionTime'] >= pit_in]
        if len(seg) >= 10:
            entry_points = [{"x": float(x), "y": float(y)} for x, y in zip(seg['X'], seg['Y'])]
            break

    exit_points = []
    for _, lap in laps[laps['PitOutTime'].notna()].iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        pit_out = lap['PitOutTime']
        if tel.empty or pd.isna(pit_out):
            continue
        seg = tel[tel['SessionTime'] <= pit_out]
        if len(seg) >= 10:
            exit_points = [{"x": float(x), "y": float(y)} for x, y in zip(seg['X'], seg['Y'])]
            break

    return {"entry": entry_points, "exit": exit_points}