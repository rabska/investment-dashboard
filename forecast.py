"""
forecast.py — LSTM price forecasting pipeline
==============================================

Architecture (per article):
  Input  : 60-day rolling window of MinMax-scaled Close prices
  LSTM 1 : 128 units, return_sequences=True
  LSTM 2 : 64  units, return_sequences=False
  Dense  : 25  neurons, ReLU activation
  Output : 1   neuron  (next-day scaled Close)

Training:
  • 85 % of history for training, 15 % held out for evaluation
  • MinMaxScaler fitted on training split only
  • Adam optimiser, MSE loss, 35 epochs, batch size 32

Test evaluation (printed to console):
  RMSE · MAE · MAPE · R²

Future prediction:
  Recursive / autoregressive: each predicted step is appended to
  the rolling seed window and fed back as the next model input,
  producing a genuine curve rather than a flat line.

Caching:
  A saved forecast is reused until the underlying price data changes
  (detected via SHA-256 hash + file mtime).  Re-training is triggered
  automatically when the cache is stale.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress TensorFlow / Keras verbosity at import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

from data_manager import DataManager


# ============================================================
# HYPER-PARAMETERS  (match the article exactly)
# ============================================================

LOOKBACK         : int   = 60      # days fed into each training / prediction step
TEST_CONTEXT     : int   = 90      # context days prepended when building test examples
TRAIN_RATIO      : float = 0.85    # fraction of history used for training
LSTM_UNITS_1     : int   = 128     # neurons in first LSTM layer
LSTM_UNITS_2     : int   = 64      # neurons in second LSTM layer
DENSE_HIDDEN     : int   = 25      # neurons in hidden Dense layer
EPOCHS           : int   = 35
BATCH_SIZE       : int   = 32

TRADING_DAYS_3MO : int   = 63      # ~3 calendar months of trading days
TRADING_DAYS_1Y  : int   = 252     # ~1 calendar year of trading days

MIN_ROWS_FOR_LSTM: int   = LOOKBACK + 30   # safety floor; flat fallback below this


# ============================================================
# MODEL FACTORY
# ============================================================

def _build_model(lookback: int):
    """
    Construct the LSTM model described in the article:
      LSTM(128, return_sequences=True)
      LSTM(64,  return_sequences=False)
      Dense(25, relu)
      Dense(1)
    Compiled with Adam + MSE.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.get_logger().setLevel("ERROR")

    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.LSTM(LSTM_UNITS_2, return_sequences=False),
        layers.Dense(DENSE_HIDDEN, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ============================================================
# FORECAST GENERATOR
# ============================================================

class ForecastGenerator:
    """
    Trains one LSTM per ticker (shared between horizons within a session)
    and writes forecast parquets that the dashboard reads at render time.
    """

    def __init__(self) -> None:
        self.dm = DataManager()
        # In-memory model cache: {ticker: (model, scaler) | None}
        # Allows reuse of the same trained model for both horizons.
        self._model_cache: dict[str, tuple | None] = {}

    # ========================================================
    # CLOSE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_close(df: pd.DataFrame) -> pd.Series:
        """
        Return a clean, tz-naive, sorted Close series from a price parquet.
        Handles both flat and MultiIndex column layouts produced by yfinance.
        """
        if df is None or df.empty:
            return pd.Series(dtype="float64")

        frame = df.copy()

        # Flatten MultiIndex if present
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                "_".join(str(x) for x in col
                         if str(x) not in {"", "nan", "None"}).strip("_")
                for col in frame.columns
            ]

        # Prefer an exact "Close" column, then any column whose name
        # contains "close" (case-insensitive) as a fallback.
        candidates = (
            ["Close"] +
            [c for c in frame.columns if c != "Close" and "close" in c.lower()]
        )

        for candidate in candidates:
            if candidate not in frame.columns:
                continue
            obj = frame[candidate]
            if isinstance(obj, pd.DataFrame):
                obj = obj.iloc[:, 0]
            s = pd.to_numeric(obj, errors="coerce").dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name  = "Close"
            return s.sort_index()

        return pd.Series(dtype="float64")

    # ========================================================
    # HORIZON -> PERIODS
    # ========================================================

    @staticmethod
    def _periods(horizon: str) -> int:
        if horizon == "3mo":
            return TRADING_DAYS_3MO
        if horizon == "1y":
            return TRADING_DAYS_1Y
        raise ValueError(f"Unknown horizon '{horizon}'. Use '3mo' or '1y'.")

    # ========================================================
    # FLAT FALLBACK
    # ========================================================

    @staticmethod
    def _flat_fallback(last_value: float, periods: int) -> np.ndarray:
        """Last-known-value flat line; used when LSTM cannot be trained."""
        return np.full(periods, last_value, dtype="float64")

    # ========================================================
    # LSTM TRAINING  (called once per ticker per session)
    # ========================================================

    def _train(self, close: pd.Series, ticker: str) -> tuple | None:
        """
        Trains the LSTM on the training split of *close* and evaluates on
        the test split.  Prints RMSE / MAE / MAPE / R² to the console.

        Returns (model, scaler) on success, None on failure.
        """
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import (mean_squared_error,
                                     mean_absolute_error,
                                     r2_score)

        values = close.values.astype("float64").reshape(-1, 1)
        n      = len(values)

        if n < MIN_ROWS_FOR_LSTM:
            print(f"  [{ticker}] Only {n} rows — need >= {MIN_ROWS_FOR_LSTM}."
                  f"  Skipping LSTM; flat fallback will be used.")
            return None

        # ── Train / test split ───────────────────────────────────
        train_end = int(n * TRAIN_RATIO)
        train_raw = values[:train_end]

        # ── MinMax scaling (fit on training data only) ───────────
        scaler   = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler.fit_transform(train_raw)

        # ── Training sequences: X = [t-60 … t-1], Y = t ─────────
        X_tr, y_tr = [], []
        for i in range(LOOKBACK, len(train_sc)):
            X_tr.append(train_sc[i - LOOKBACK : i, 0])
            y_tr.append(train_sc[i, 0])

        if len(X_tr) < 10:
            print(f"  [{ticker}] Not enough training sequences. "
                  f"Flat fallback will be used.")
            return None

        X_tr = np.array(X_tr).reshape(-1, LOOKBACK, 1)
        y_tr = np.array(y_tr)

        # ── Train ────────────────────────────────────────────────
        print(f"  [{ticker}] Training LSTM "
              f"({train_end} train rows | {n - train_end} test rows) …")

        model = _build_model(LOOKBACK)
        model.fit(
            X_tr, y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        # ── Test evaluation ──────────────────────────────────────
        # Build test sequences using up to TEST_CONTEXT real days as
        # context so the model has a full LOOKBACK window even at the
        # very first test step.
        ctx_start   = max(0, train_end - TEST_CONTEXT)
        ctx_raw     = values[ctx_start:]        # context rows + test rows (raw)
        ctx_sc      = scaler.transform(ctx_raw)

        context_len = train_end - ctx_start     # rows that are pre-test context

        X_te, y_te_raw = [], []
        for i in range(context_len, len(ctx_sc)):
            if i < LOOKBACK:                    # guard: need full window
                continue
            X_te.append(ctx_sc[i - LOOKBACK : i, 0])
            # Corresponding raw (unscaled) truth value
            raw_idx = i - context_len           # index into ctx_raw test portion
            y_te_raw.append(ctx_raw[context_len + raw_idx, 0])

        if X_te:
            X_te       = np.array(X_te).reshape(-1, LOOKBACK, 1)
            y_pred_sc  = model.predict(X_te, verbose=0)
            y_pred     = scaler.inverse_transform(y_pred_sc).flatten()
            y_true     = np.array(y_te_raw).flatten()

            # ── Metrics ──────────────────────────────────────────
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            mask = y_true != 0
            mape = (
                float(np.mean(
                    np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                ) * 100)
                if mask.any() else float("nan")
            )
            r2 = float(r2_score(y_true, y_pred))

            print(
                f"  [{ticker}] Test metrics — "
                f"RMSE: {rmse:.4f} | "
                f"MAE: {mae:.4f} | "
                f"MAPE: {mape:.2f}% | "
                f"R²: {r2:.4f}"
            )
        else:
            print(f"  [{ticker}] Test window too small — metrics unavailable.")

        return model, scaler

    # ========================================================
    # RECURSIVE FUTURE PREDICTION  (produces a genuine curve)
    # ========================================================

    def _predict_future(
        self,
        close  : pd.Series,
        model,
        scaler,
        periods: int,
    ) -> np.ndarray:
        """
        Autoregressively predicts `periods` future steps.

        Seed  = the last LOOKBACK observed Close prices, scaled via the
                same scaler that was fitted during training.
        Loop  = each newly predicted value is appended to the rolling
                window so the model always receives a fresh LOOKBACK-
                length context — producing an evolving price curve.
        """
        seed_raw    = close.values[-LOOKBACK:].reshape(-1, 1).astype("float64")
        seed_scaled = scaler.transform(seed_raw).flatten().tolist()

        future_scaled: list[float] = []
        for _ in range(periods):
            window = np.array(seed_scaled[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
            pred_s = float(model.predict(window, verbose=0)[0, 0])
            future_scaled.append(pred_s)
            seed_scaled.append(pred_s)

        future_prices = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()

        return future_prices

    # ========================================================
    # GET-OR-TRAIN  (cache so both horizons share one fit)
    # ========================================================

    def _get_model(self, close: pd.Series, ticker: str) -> tuple | None:
        """Return cached (model, scaler), training if not yet done."""
        if ticker not in self._model_cache:
            self._model_cache[ticker] = self._train(close, ticker)
        return self._model_cache[ticker]

    # ========================================================
    # BUILD ONE FORECAST DATAFRAME
    # ========================================================

    def build_forecast_frame(self, ticker: str, horizon: str) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [Open, High, Low, Close, Volume]
        indexed by future business dates.

        • Close  = LSTM recursive predictions (or flat fallback).
        • Open / High / Low  = same as Close so the dashboard's
          pct_change() and _extract_close_series() work correctly.
        • Volume = 0.
        """
        price_path = self.dm.get_price_path(ticker)
        if not price_path.exists():
            print(f"  [{ticker}] No local price file — skipping forecast.")
            return pd.DataFrame()

        df    = pd.read_parquet(price_path)
        close = self._extract_close(df)
        if close.empty:
            print(f"  [{ticker}] Empty Close series — skipping forecast.")
            return pd.DataFrame()

        periods  = self._periods(horizon)
        last_val = float(close.iloc[-1])

        # ── LSTM (or flat fallback) ──────────────────────────────
        cache = self._get_model(close, ticker)

        if cache is not None:
            model, scaler = cache
            try:
                preds = self._predict_future(close, model, scaler, periods)
            except Exception as exc:
                print(f"  [{ticker}] Prediction error ({exc}) — flat fallback.")
                preds = self._flat_fallback(last_val, periods)
        else:
            preds = self._flat_fallback(last_val, periods)

        # ── Future business-day date index ───────────────────────
        last_date    = close.index.max()
        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1),
            periods=periods,
        )

        # ── Assemble OHLCV frame ─────────────────────────────────
        forecast_df = pd.DataFrame(
            {
                "Open"  : preds,
                "High"  : preds,
                "Low"   : preds,
                "Close" : preds,
                "Volume": np.zeros(periods, dtype="float64"),
            },
            index=future_dates,
        )
        forecast_df.index.name = "Date"

        return forecast_df

    # ========================================================
    # CACHE VALIDATION
    # ========================================================

    def _forecast_is_valid(self, ticker: str, horizon: str) -> bool:
        """
        A saved forecast is reused only when:
          • the forecast parquet exists
          • its metadata parquet exists
          • stored price_hash matches the current file hash
          • stored price_mtime matches the current file mtime
          • the horizon tag matches
        Any mismatch triggers a full retrain + re-save.
        """
        fcast_path  = self.dm.get_forecast_path(ticker, horizon)
        meta        = self.dm.load_forecast_metadata(ticker, horizon)
        price_state = self.dm.get_price_state(ticker)

        if not fcast_path.exists() or meta is None or price_state is None:
            return False

        return (
            str(meta.get("price_hash",  ""))   == str(price_state["price_hash"])
            and float(meta.get("price_mtime", -1)) == float(price_state["price_mtime"])
            and str(meta.get("horizon",     ""))   == horizon
        )

    # ========================================================
    # PERSIST ONE FORECAST
    # ========================================================

    def _save_forecast(
        self,
        ticker     : str,
        horizon    : str,
        forecast_df: pd.DataFrame,
    ) -> None:
        path = self.dm.get_forecast_path(ticker, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_parquet(path)

        price_state = self.dm.get_price_state(ticker)
        metadata = {
            "ticker"        : ticker,
            "horizon"       : horizon,
            "training_date" : datetime.now().isoformat(timespec="seconds"),
            "method"        : "lstm",
            "lookback"      : LOOKBACK,
            "epochs"        : EPOCHS,
            "batch_size"    : BATCH_SIZE,
            "lstm_units_1"  : LSTM_UNITS_1,
            "lstm_units_2"  : LSTM_UNITS_2,
            "dense_hidden"  : DENSE_HIDDEN,
            "train_ratio"   : TRAIN_RATIO,
            "periods"       : len(forecast_df),
            "last_real_date": (
                pd.to_datetime(price_state["last_date"]).isoformat()
                if price_state else None
            ),
            "price_hash"    : price_state["price_hash"]  if price_state else None,
            "price_mtime"   : price_state["price_mtime"] if price_state else None,
        }
        self.dm.save_forecast_metadata(ticker, horizon, metadata)
        print(f"  [{ticker}] Saved forecast ({horizon}) — {len(forecast_df)} rows.")

    # ========================================================
    # PUBLIC: GENERATE ONE FORECAST
    # ========================================================

    def generate_forecast(self, ticker: str, horizon: str) -> pd.DataFrame:
        """Force-rebuild a forecast regardless of cache state."""
        forecast_df = self.build_forecast_frame(ticker, horizon)
        if forecast_df.empty:
            return pd.DataFrame()
        self._save_forecast(ticker, horizon, forecast_df)
        return forecast_df

    def generate_forecast_if_needed(
        self, ticker: str, horizon: str
    ) -> pd.DataFrame:
        """Rebuild only when the cache is missing or stale."""
        if self._forecast_is_valid(ticker, horizon):
            path = self.dm.get_forecast_path(ticker, horizon)
            print(f"  [{ticker}] Forecast ({horizon}) cache valid — skipping retrain.")
            return pd.read_parquet(path)
        return self.generate_forecast(ticker, horizon)

    # ========================================================
    # PUBLIC: GENERATE ALL TICKERS x HORIZONS
    # ========================================================

    def generate_forecasts(self, tickers: list[str]) -> None:
        """
        Iterates over every ticker and both horizons.

        The LSTM is trained once per ticker (weights are reused for both
        the 3-month and 1-year rollouts; only the number of recursive
        prediction steps differs).
        """
        for ticker in tickers:
            # Evict any stale model so a fresh one is trained for this ticker
            self._model_cache.pop(ticker, None)

            for horizon in ["3mo", "1y"]:
                try:
                    self.generate_forecast_if_needed(ticker, horizon)
                except Exception as exc:
                    print(f"  [{ticker}] ERROR ({horizon}): {exc}")

