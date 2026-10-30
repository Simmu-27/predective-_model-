import json, math, re
from datetime import datetime
from pathlib import Path
import joblib
from flask import Flask, jsonify, render_template, request
from features import MIN_HISTORY, features_for_date

BASE = Path(__file__).parent
app = Flask(__name__)
BUNDLE, ANALYTICS, LOAD_ERROR = None, None, None
try:
    BUNDLE = joblib.load(BASE / "models/weather_model.joblib")
    ANALYTICS = json.loads((BASE / "models/analytics.json").read_text())
except Exception as e:
    LOAD_ERROR = f"Model/analytics not loaded ({e}). Run: python train_model.py"

def err(msg, code=400): return jsonify({"success": False, "error": msg}), code

@app.get("/")
def home(): return render_template("index.html", min_history=MIN_HISTORY)

@app.get("/api/health")
def health():
    return jsonify({"success": True, "status": "ok", "model_loaded": BUNDLE is not None, "detail": LOAD_ERROR})

@app.get("/api/analytics")
def analytics():
    if ANALYTICS is None: return err(LOAD_ERROR, 503)
    return jsonify({"success": True, **ANALYTICS})

@app.get("/api/model-metrics")
def metrics():
    if BUNDLE is None: return err(LOAD_ERROR, 503)
    return jsonify({"success": True, "metrics": BUNDLE["metrics"], "best_model": BUNDLE["best"],
                    "cv_mae": BUNDLE["cv_mae"], "test_period": BUNDLE["test_period"]})

@app.post("/api/predict")
def predict():
    if BUNDLE is None: return err(LOAD_ERROR, 503)
    data = request.get_json(silent=True)
    if not isinstance(data, dict): return err("Request body must be JSON.")
    try:
        date = datetime.strptime(str(data.get("date", "")), "%Y-%m-%d")
    except ValueError:
        return err("'date' must be in YYYY-MM-DD format.")
    hist = data.get("history")
    if isinstance(hist, str): hist = [x for x in re.split(r"[,\s;]+", hist.strip()) if x]
    if not isinstance(hist, list): return err("'history' must be a list of daily temperatures.")
    try: hist = [float(x) for x in hist]
    except (TypeError, ValueError): return err("'history' must contain only numbers.")
    if len(hist) < MIN_HISTORY:
        return err(f"Need at least {MIN_HISTORY} daily temperatures (oldest to newest, ending the day before the date); got {len(hist)}.")
    if any(not math.isfinite(v) or not -60 <= v <= 60 for v in hist): return err("All temperatures must be between -60 and 60 °C.")
    name = data.get("model") or BUNDLE["best"]
    if name not in BUNDLE["models"]: return err(f"Unknown model. Choose from {list(BUNDLE['models'])}.")
    pred = float(BUNDLE["models"][name].predict(features_for_date(hist, date)[BUNDLE["features"]])[0])
    return jsonify({"success": True, "prediction_c": round(pred, 2), "date": data["date"], "model": name})

if __name__ == "__main__":
    app.run(debug=True)
