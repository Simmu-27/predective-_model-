"""Feature engineering shared by training AND the Flask app (avoids train/serve mismatch).
Every feature for day t uses only data up to day t-1 (no leakage)."""
import numpy as np
import pandas as pd

FEATURES = ["lag_1", "lag_2", "lag_7", "roll_3", "roll_7", "roll_30",
            "std_7", "trend", "month", "doy_sin", "doy_cos"]
MIN_HISTORY = 30  # days of recent temperatures needed to predict one day


def build_features(s: pd.Series) -> pd.DataFrame:
    """s: daily temperature series with a DatetimeIndex (consecutive days)."""
    p = s.shift(1)  # everything below is based on yesterday and earlier
    d = pd.DataFrame(index=s.index)
    d["lag_1"], d["lag_2"], d["lag_7"] = s.shift(1), s.shift(2), s.shift(7)
    d["roll_3"], d["roll_7"], d["roll_30"] = (p.rolling(n).mean() for n in (3, 7, 30))
    d["std_7"] = p.rolling(7).std()
    d["trend"] = d["lag_1"] - d["roll_7"]
    d["month"] = d.index.month
    doy = d.index.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return d[FEATURES]


def features_for_date(history, date) -> pd.DataFrame:
    """history: last >=30 daily temps (oldest -> newest), ending the day BEFORE `date`."""
    h = [float(x) for x in history][-MIN_HISTORY:]
    idx = pd.date_range(end=pd.Timestamp(date), periods=len(h) + 1, freq="D")
    return build_features(pd.Series(h + [np.nan], index=idx)).iloc[[-1]]
