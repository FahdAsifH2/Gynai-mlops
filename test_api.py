import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "    AgeAtStartOfSpell": 22,
    "Body Mass Index at Booking": 21,
    "WeightMeasured": 50,
    "Height": 155,
    "Parity": 0,
    "Gravida": 0,
    "No_Of_previous_Csections": 0,
    "Gestation (Days)": 260,
    "Gestation at booking (Weeks)": 27.0,
    "Obese_Encoded": 0,
    "GestationalDiabetes_Encoded": 0,

    "Ethnicity_WEU": 0,
    "Ethnicity_GBR": 0,
    "Ethnicity_OTH": 1,
    "Ethnicity_NAF": 0,
    "Ethnicity_MEA": 0
}



print(f"Sending request with {len(data)} features...")
response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")