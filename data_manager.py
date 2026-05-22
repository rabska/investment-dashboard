from pathlib import Path
from datetime import timedelta
import hashlib

import pandas as pd
import yfinance as yf


# ============================================================
# BASE PATHS
# ============================================================

BASE_DATA_DIR = Path(r"D:\0Storage\bachelors\investment-dashboard\data")

PRICE_DIR = BASE_DATA_DIR / "prices"
INFO_DIR  = BASE_DATA_DIR / "asset_info"

FORECAST_PRICE_DIR = BASE_DATA_DIR / "forecast_prices"
FORECAST_1MO_DIR   = FORECAST_PRICE_DIR / "1mo"   # replaces "1y"
FORECAST_3MO_DIR   = FORECAST_PRICE_DIR / "3mo"
FORECAST_META_DIR  = BASE_DATA_DIR / "forecast_meta"

for folder in [
    PRICE_DIR,
    INFO_DIR,
    FORECAST_1MO_DIR,
    FORECAST_3MO_DIR,
    FORECAST_META_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Canonical set of supported horizons
SUPPORTED_HORIZONS: frozenset[str] = frozenset({"1mo", "3mo"})


# ============================================================
# DATA MANAGER
# ============================================================

class DataManager:

    def __init__(self):
        pass

    # ========================================================
    # SMALL HELPERS
    # ========================================================

    @staticmethod
    def _clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize index, sort, and remove duplicates."""
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    @staticmethod
    def _extract_close_series(df: pd.DataFrame) -> pd.Series | None:
        """
        Safe Close extraction for metadata/hash purposes.
        Does not affect load_prices().
        """
        if df is None or df.empty:
            return None

        frame = df.copy()

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                "_".join([str(x) for x in col
                          if str(x) not in {"", "nan", "None"}]).strip("_")
                for col in frame.columns.to_list()
            ]

        if "Close" in frame.columns:
            close_obj = frame.loc[:, "Close"]
            if isinstance(close_obj, pd.DataFrame):
                close_obj = close_obj.iloc[:, 0]
            if isinstance(close_obj, pd.Series):
                s = pd.to_numeric(close_obj, errors="coerce").dropna()
                if not s.empty:
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s.name  = "Close"
                    return s.sort_index()

        close_candidates = [c for c in frame.columns if "close" in str(c).lower()]
        for col in close_candidates:
            obj = frame.loc[:, col]
            if isinstance(obj, pd.DataFrame):
                obj = obj.iloc[:, 0]
            if isinstance(obj, pd.Series):
                s = pd.to_numeric(obj, errors="coerce").dropna()
                if not s.empty:
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s.name  = str(col)
                    return s.sort_index()

        return None

    @staticmethod
    def _hash_close_series(close_series: pd.Series) -> str:
        """Stable SHA-256 hash of a Close series.  Used to detect stale forecasts."""
        s = pd.to_numeric(close_series, errors="coerce").dropna().copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        payload = pd.util.hash_pandas_object(s, index=True).values.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def get_price_path(self, ticker: str) -> Path:
        """Forecast pipeline expects this method."""
        return PRICE_DIR / f"{ticker}.parquet"

    def get_price_state(self, ticker: str) -> dict | None:
        """Small state snapshot used by forecast caching."""
        price_path = self.get_price_path(ticker)
        if not price_path.exists():
            return None
        try:
            df = pd.read_parquet(price_path)
        except Exception:
            return None

        df    = self._clean_price_frame(df)
        close = self._extract_close_series(df)
        if close is None or close.empty:
            return None

        return {
            "ticker"     : ticker,
            "last_date"  : close.index.max(),
            "price_hash" : self._hash_close_series(close),
            "price_mtime": price_path.stat().st_mtime,
            "price_path" : str(price_path),
        }

    @staticmethod
    def _download_history(ticker: str, start_date: str | None = None) -> pd.DataFrame:
        """Download daily history from Yahoo Finance."""
        df = yf.download(
            ticker,
            start=start_date,
            auto_adjust=True,
            progress=False,
            interval="1d",
        )
        return DataManager._clean_price_frame(df)

    @staticmethod
    def _download_full_history(ticker: str) -> pd.DataFrame:
        df = yf.download(
            ticker,
            period="max",
            auto_adjust=True,
            progress=False,
            interval="1d",
        )
        return DataManager._clean_price_frame(df)

    # ========================================================
    # FORECAST PATHS / METADATA
    # ========================================================

    @staticmethod
    def get_forecast_path(ticker: str, horizon: str) -> Path:
        """
        Returns the parquet path for a forecast file.

        Supported horizons
        ------------------
        "1mo"  →  ~1 calendar month   (21 trading days)
        "3mo"  →  ~3 calendar months  (63 trading days)
        """
        if horizon == "1mo":
            return FORECAST_1MO_DIR / f"{ticker}.parquet"
        if horizon == "3mo":
            return FORECAST_3MO_DIR / f"{ticker}.parquet"
        raise ValueError(
            f"horizon must be '1mo' or '3mo', got '{horizon}'"
        )

    @staticmethod
    def get_forecast_meta_path(ticker: str, horizon: str) -> Path:
        return FORECAST_META_DIR / f"{ticker}_{horizon}_meta.parquet"

    def save_forecast_metadata(self, ticker: str, horizon: str,
                                metadata: dict) -> None:
        path = self.get_forecast_meta_path(ticker, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metadata]).to_parquet(path)

    def load_forecast_metadata(self, ticker: str, horizon: str) -> dict | None:
        path = self.get_forecast_meta_path(ticker, horizon)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            return df.iloc[0].to_dict()
        except Exception:
            return None

    def invalidate_forecasts(self, ticker: str) -> None:
        """Delete forecast files + metadata for all supported horizons."""
        paths = []
        for horizon in SUPPORTED_HORIZONS:
            paths.append(self.get_forecast_path(ticker, horizon))
            paths.append(self.get_forecast_meta_path(ticker, horizon))

        for path in paths:
            if path.exists():
                path.unlink()
                print(f"Deleted outdated forecast artifact: {path.name}")

    def invalidate_all_forecasts(self, tickers: list[str]) -> None:
        """Use when any real data changed and a full forecast rebuild is needed."""
        for ticker in tickers:
            self.invalidate_forecasts(ticker)

    # ========================================================
    # UPDATE SINGLE TICKER
    # ========================================================

    def update_ticker(self, ticker: str) -> tuple[pd.DataFrame, bool]:
        """
        Returns:
            (prices_df, changed)

        changed=True only if local data was actually updated.
        """
        print(f"\nUpdating {ticker}...")

        price_path = self.get_price_path(ticker)
        info_path  = INFO_DIR / f"{ticker}_info.parquet"

        changed = False
        prices  = pd.DataFrame()

        # ── Load existing data ───────────────────────────────────
        if price_path.exists():
            existing = pd.read_parquet(price_path)
            existing = self._clean_price_frame(existing)

            if existing.empty:
                print("Existing file is empty. Downloading full history...")
                prices = self._download_full_history(ticker)
                if not prices.empty:
                    prices.to_parquet(price_path)
                    changed = True
                    print(f"Saved {len(prices)} rows.")
                else:
                    print("No data returned from Yahoo Finance.")

            else:
                last_date = existing.index.max()
                reload_buffer_days = 14
                start_date = (
                    last_date - timedelta(days=reload_buffer_days)
                ).strftime("%Y-%m-%d")

                print("Existing data found.")
                print(f"Last local date: {last_date.date()}")
                print(f"Refreshing recent window from: {start_date}")

                new_data = self._download_history(
                    ticker=ticker, start_date=start_date
                )

                if new_data.empty:
                    print("No refreshed rows returned from Yahoo Finance.")
                    prices = existing
                else:
                    combined = pd.concat([existing, new_data], axis=0)
                    combined = self._clean_price_frame(combined)

                    if combined.equals(existing):
                        print("All data is up to date.")
                        prices = existing
                    else:
                        prices = combined
                        prices.to_parquet(price_path)
                        changed = True
                        added_rows = max(0, len(combined) - len(existing))
                        if added_rows > 0:
                            print(f"Loaded {added_rows} new or refreshed rows.")
                        else:
                            print("Refreshed recent rows.")

        # ── New asset ────────────────────────────────────────────
        else:
            print("No local data found. Downloading full history...")
            prices = self._download_full_history(ticker)
            if prices.empty:
                print("No data returned from Yahoo Finance.")
            else:
                prices.to_parquet(price_path)
                changed = True
                print(f"Saved {len(prices)} rows.")

        # ── Save asset info ──────────────────────────────────────
        if not info_path.exists():
            try:
                info    = yf.Ticker(ticker).info
                info_df = pd.DataFrame([info])
                info_df.to_parquet(info_path)
                print("Asset info saved.")
            except Exception as e:
                print(f"Info download failed: {e}")

        # ── Invalidate forecasts if data changed ─────────────────
        if changed:
            self.invalidate_forecasts(ticker)

        return prices, changed

    # ========================================================
    # UPDATE MULTIPLE TICKERS
    # ========================================================

    def update_tickers(self, tickers: list[str]) -> list[str]:
        changed_tickers = []
        for ticker in tickers:
            try:
                _, changed = self.update_ticker(ticker)
                if changed:
                    changed_tickers.append(ticker)
            except Exception as e:
                print(f"\nERROR: {ticker}")
                print(e)

        if changed_tickers:
            print("\nUpdated tickers: " + ", ".join(changed_tickers))
        else:
            print("\nAll tickers are up to date.")

        return changed_tickers

    # ========================================================
    # LOAD COMBINED CLOSE PRICES
    # ========================================================

    def load_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Historical prices only."""
        all_prices = []

        for ticker in tickers:
            path = PRICE_DIR / f"{ticker}.parquet"
            if not path.exists():
                print(f"Missing local data: {ticker}")
                continue

            df = pd.read_parquet(path)
            if df.empty or "Close" not in df.columns:
                continue

            series       = df["Close"].copy()
            series.name  = ticker
            series.index = pd.to_datetime(series.index).tz_localize(None)
            all_prices.append(series)

        if not all_prices:
            return pd.DataFrame()

        combined       = pd.concat(all_prices, axis=1).sort_index()
        combined.index = pd.to_datetime(combined.index).tz_localize(None)
        return combined