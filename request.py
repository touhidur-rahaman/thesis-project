import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'area':2, 'type':1, 'bed':3,'bath':3, 'sqft':1300})

print(r.json())