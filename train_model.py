"""Train WeatherWise models.
Usage: python train_model.py --csv data/weather.csv --date-col date --target-col temperature"""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES, build_features

def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/weather.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target-col", default="temperature")
    ap.add_argument("--test-size", type=float, default=0.2)
    a = ap.parse_args()
    if not Path(a.csv).exists():
        sys.exit(f"Dataset not found: {a.csv}. See README 'Dataset' section.")

    df = pd.read_csv(a.csv)
    print("Shape:", df.shape); print(df.head()); print(df.dtypes)
    if a.date_col not in df or a.target_col not in df:
        sys.exit(f"Columns {a.date_col!r}/{a.target_col!r} not found. Available: {list(df.columns)}")

    # 1) Cleaning
    df = df[[a.date_col, a.target_col]].copy()
    df[a.date_col] = pd.to_datetime(df[a.date_col], errors="coerce")
    df[a.target_col] = pd.to_numeric(df[a.target_col], errors="coerce")
    n0 = len(df)
    df = df.dropna().drop_duplicates()
    df = df[(df[a.target_col] > -90) & (df[a.target_col] < 60)]
    print(f"Removed {n0 - len(df)} missing/duplicate/invalid rows")
    s = df.groupby(df[a.date_col].dt.normalize())[a.target_col].mean().sort_index()
    s = s.asfreq("D").interpolate(limit=3)  # only short gaps are filled

    # 2) Features + chronological split
    d = build_features(s)
    d["target"] = s
    d = d.dropna()
    if len(d) < 365: sys.exit("Need at least ~1 year of daily data.")
    cut = int(len(d) * (1 - a.test_size))
    tr, te = d.iloc[:cut], d.iloc[cut:]
    print(f"Train {tr.index.min().date()}..{tr.index.max().date()} | Test {te.index.min().date()}..{te.index.max().date()}")

    # 3) Models, tuned with time-ordered cross-validation on the TRAIN part only
    cv = TimeSeriesSplit(n_splits=5)
    specs = {
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), {}),
        "Ridge Regression": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                          {"max_depth": [None, 10], "min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=42),
                              {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}),
    }
    models, results, cv_mae, params = {}, {"Persistence baseline (yesterday)": metrics(te["target"], te["lag_1"])}, {}, {}
    for name, (est, grid) in specs.items():
        gs = GridSearchCV(est, grid or {}, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(tr[FEATURES], tr["target"])
        models[name], cv_mae[name], params[name] = gs.best_estimator_, float(-gs.best_score_), gs.best_params_
        results[name] = metrics(te["target"], gs.predict(te[FEATURES]))
        print(f"{name}: CV MAE {cv_mae[name]:.3f} | test {results[name]} | params {gs.best_params_}")
    print("Persistence:", results["Persistence baseline (yesterday)"])
    best = min(cv_mae, key=cv_mae.get)  # chosen on validation, NOT on the test set

    # 4) Analytics for the dashboard
    monthly = s.resample("MS").mean().dropna()
    clim = s.groupby(s.index.month).mean()
    corr = d.corr()["target"].drop("target").round(3)
    analytics = {
        "summary": {"records": int(len(s)), "start": str(s.index.min().date()), "end": str(s.index.max().date()),
                    "mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())},
        "monthly_trend": {"labels": [x.strftime("%Y-%m") for x in monthly.index], "values": monthly.round(2).tolist()},
        "monthly_average": {"labels": list(range(1, 13)), "values": clim.round(2).reindex(range(1, 13)).tolist()},
        "correlation": {"labels": corr.index.tolist(), "values": corr.tolist()},
    }
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"models": models, "best": best, "features": FEATURES, "metrics": results, "cv_mae": cv_mae,
                 "best_params": {k: {p: (v if v is None else float(v) if isinstance(v, (float, np.floating)) else v) for p, v in d_.items()} for k, d_ in params.items()},
                 "test_period": [str(te.index.min().date()), str(te.index.max().date())]},
                "models/weather_model.joblib")
    Path("models/analytics.json").write_text(json.dumps(analytics))
    print("Best model (lowest CV MAE):", best, "| saved to models/")

if __name__ == "__main__":
    main()
