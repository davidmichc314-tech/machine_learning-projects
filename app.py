import os
import pickle
from pathlib import Path

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

mapping_dict = {"income": {0: "<=50K", 1: ">50K"}}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/result", methods=["POST"])
def result():
    try:
        features = [
            int(request.form["age"]),
            int(request.form["w_class"]),
            int(request.form["edu"]),
            int(request.form["martial_stat"]),
            int(request.form["occup"]),
            int(request.form["relation"]),
            int(request.form["gender"]),
            int(request.form["race"]),
            int(request.form["c_gain"]),
            int(request.form["c_loss"]),
            int(request.form["hours_per_week"]),
            int(request.form["native-country"]),
        ]

        prediction = model.predict([features])[0]
        predicted_income = mapping_dict["income"][prediction]

        return render_template("index.html", prediction_text=f"Predicted Income: {predicted_income}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)