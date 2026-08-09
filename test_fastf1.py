import fastf1

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2026, 'Silverstone', 'R')
session.load()

print(session.laps.head())

fastest_lap = session.laps.pick_driver('NOR').pick_fastest()
telemetry = fastest_lap.get_telemetry()
print(telemetry[['Speed', 'Throttle', 'Brake', 'nGear', 'RPM']].head(20))