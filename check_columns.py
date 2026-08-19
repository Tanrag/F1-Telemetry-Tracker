import fastf1

fastf1.Cache.enable_cache('cache')

print(fastf1.__version__)

session = fastf1.get_session(2026, 'Silverstone', 'R')
session.load()
lap = session.laps.pick_drivers('NOR').iloc[5]
telemetry = lap.get_telemetry()

print(telemetry.columns.tolist())
print(telemetry['Status'].unique())
print(telemetry['DRS'].unique())