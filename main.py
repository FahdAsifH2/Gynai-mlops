from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("models/model.pkl")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return "GynAI API is running fine"

def engineer_features(data):
    age = data['    AgeAtStartOfSpell']
    if age <= 20:
        data['Age_Risk'] = 0
    elif age <= 35:
        data['Age_Risk'] = 1
    else:
        data['Age_Risk'] = 2
    
    bmi = data['Body Mass Index at Booking']
    if bmi < 18.5:
        data['BMI_Category'] = 0
    elif bmi < 25:
        data['BMI_Category'] = 1
    elif bmi < 30:
        data['BMI_Category'] = 2
    else:
        data['BMI_Category'] = 3
    
    data['Prev_Delivery_Risk'] = data['No_Of_previous_Csections'] + (1 if data['Parity'] > 3 else 0)
    
    data['High_Risk_Score'] = (
        (data['Obese_Encoded'] * 2) + 
        (data['GestationalDiabetes_Encoded'] * 2) + 
        (1 if data['No_Of_previous_Csections'] > 0 else 0) + 
        (1 if data['    AgeAtStartOfSpell'] > 35 else 0)
    )
    
    gestation = data['Gestation (Days)']
    if gestation <= 259:
        data['Gestation_Risk'] = 0
    elif gestation <= 280:
        data['Gestation_Risk'] = 1
    else:
        data['Gestation_Risk'] = 2
    
    data['Weight_Height_Ratio'] = data['WeightMeasured'] / data['Height']
    
    return data

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        required_features = [
            "    AgeAtStartOfSpell",
            "Body Mass Index at Booking",
            "WeightMeasured",
            "Height",
            "Parity",
            "Gravida",
            "No_Of_previous_Csections",
            "Gestation (Days)",
            "Gestation at booking (Weeks)",
            "Obese_Encoded",
            "GestationalDiabetes_Encoded",
            "Ethnicity_WEU",
            "Ethnicity_GBR",
            "Ethnicity_OTH",
            "Ethnicity_NAF",
            "Ethnicity_MEA"
        ]

        missing = [f for f in required_features if f not in input_data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        data = {k: float(v) for k, v in input_data.items()}
        data = engineer_features(data)
        
        feature_order = [
            "    AgeAtStartOfSpell",
            "WeightMeasured",
            "Height",
            "Body Mass Index at Booking",
            "Parity",
            "Gravida",
            "Gestation (Days)",
            "Gestation at booking (Weeks)",
            "No_Of_previous_Csections",
            "Obese_Encoded",
            "GestationalDiabetes_Encoded",
            "Ethnicity_WEU",
            "Ethnicity_GBR",
            "Ethnicity_OTH",
            "Ethnicity_NAF",
            "Ethnicity_MEA",
            "Age_Risk",
            "BMI_Category",
            "Prev_Delivery_Risk",
            "High_Risk_Score",
            "Gestation_Risk",
            "Weight_Height_Ratio",
        ]
        
        df = pd.DataFrame([[data[f] for f in feature_order]], columns=feature_order)
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        return jsonify({
            "prediction": int(prediction[0]),
            "prediction_probability": {
                "normal_delivery": float(prediction_proba[0][0]),
                "c_section": float(prediction_proba[0][1])
            },
            "delivery_mode": "C-Section" if prediction[0] == 1 else "Normal Delivery"
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)