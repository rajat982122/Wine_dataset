from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("wine.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Arrange features in correct order
        features = [
            float(data["fixed_acidity"]),
            float(data["volatile_acidity"]),
            float(data["citric_acid"]),
            float(data["residual_sugar"]),
            float(data["chlorides"]),
            float(data["free_sulfur_dioxide"]),
            float(data["total_sulfur_dioxide"]),
            float(data["density"]),
            float(data["pH"]),
            float(data["sulphates"]),
            float(data["alcohol"])
        ]

        # Scale features
        scaled_features = scaler.transform([features])

        # Predict quality
        predicted_quality = model.predict(scaled_features)[0]

        return jsonify({"predicted_quality": float(predicted_quality)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
