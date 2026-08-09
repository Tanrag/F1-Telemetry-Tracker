import requests

r = requests.get('https://api.openf1.org/v1/car_data', params={'driver_number': 1, 'session_key': 'latest'})
print(r.json()[:5])