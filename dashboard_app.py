from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# COLORS
# ============================================================

COLOR_PALETTE = np.array([
    [173, 216, 230],  # light blue (soft sky / calm baseline)
    [0, 191, 255],    # deep sky blue (liquidity / growth)
    [30, 144, 255],   # dodger blue (equities momentum)
    [0, 0, 255],      # pure blue (core market / SPY-like reference)
    [0, 0, 139],      # dark blue (defensive equities / risk-off)
    [72, 61, 139],    # dark slate blue (stability layer)
    [123, 104, 238],  # medium purple (balanced risk)
    [138, 43, 226],   # blue violet (growth + volatility)
    [128, 0, 128],    # purple (speculative assets)
    [218, 112, 214],  # orchid (high volatility / alternatives)
    [255, 0, 255],    # magenta (quant / leveraged exposure)
    [255, 20, 147],   # deep pink (aggressive growth)
    [176, 48, 96],    # muted crimson (risk / drawdowns)
    [220, 20, 60],    # crimson (market stress / equities drawdown)
    [240, 128, 128],  # light coral (soft risk exposure)
    [255, 69, 0],     # orange red (commodities / inflation shock)
    [255, 165, 0],    # orange (energy / cyclicals)
    [244, 164, 96],   # sandy brown (real assets / REITs)
    [240, 230, 140],  # khaki (defensive macro / bonds proxy)
    [128, 128, 0],    # olive (macro hedge / commodities blend)
    [139, 69, 19],    # saddle brown (real estate / hard assets)
    [255, 255, 0],    # yellow (inflation / risk alert)
    [154, 205, 50],   # yellow green (TIPS / inflation protection)
    [124, 252, 0],    # lawn green (growth / dividends)
    [144, 238, 144],  # light green (stable equity income)
    [143, 188, 143],  # dark sea green (defensive equities)
    [34, 139, 34],    # forest green (dividend growth core)
    [0, 255, 127],    # spring green (momentum / risk-on)
    [0, 255, 255],    # cyan (liquidity / bonds inverse signal)
    [0, 139, 139],    # dark cyan (fixed income / hedge layer)
    [128, 128, 128],  # gray (neutral / cash equivalent)
])

# Table highlight colours
# BLUE_STD  = meets the academic/industry standard threshold (deep blue)
# BLUE_GOOD = sufficiently good by expert assessment / better than SPY (light blue)
BLUE_STD  = "#1D4ED8"   # deep blue  – meets metric standard
BLUE_GOOD = "#93C5FD"   # light blue – sufficiently good / better than SPY
TEXT_STD  = "#FFFFFF"
TEXT_GOOD = "#1E3A5F"

# Keep legacy aliases so nothing else breaks
BLUE_BEST = BLUE_STD
TEXT_BEST = TEXT_STD


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def get_palette(n: int) -> list[str]:
    return [rgb_to_hex(COLOR_PALETTE[i % len(COLOR_PALETTE)]) for i in range(n)]


# ============================================================
# DATA HELPERS
# ============================================================

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")

    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert(None)

    out.index = idx
    out = out.loc[~out.index.isna()]
    out = out.sort_index()
    return out


def _year_bounds(prices_df: pd.DataFrame) -> tuple[int, int]:
    idx = pd.to_datetime(prices_df.index, errors="coerce")
    idx = idx[~pd.isna(idx)]
    return int(idx.min().year), int(idx.max().year)


def _default_year_range(prices_df: pd.DataFrame) -> list[int]:
    min_year, max_year = _year_bounds(prices_df)
    start_year = max(min_year, max_year - 10)
    return [start_year, max_year]


def _build_year_marks(min_year: int, max_year: int) -> dict[int, str]:
    span = max_year - min_year

    if span <= 8:
        step = 1
    elif span <= 15:
        step = 2
    elif span <= 30:
        step = 3
    else:
        step = 5

    marks = {year: str(year) for year in range(min_year, max_year + 1, step)}
    if max_year not in marks:
        marks[max_year] = str(max_year)

    return marks


def _normalize_growth(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col].dropna()

        if s.empty:
            out[col] = pd.NA
            continue

        base = float(s.iloc[0])

        if base == 0:
            out[col] = pd.NA
            continue

        out[col] = df[col] / base * 100.0

    return out


def _filter_selected_prices(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
    min_non_nan_ratio: float = 0.7,
) -> pd.DataFrame:
    prices_df = _ensure_datetime_index(prices_df)

    available_assets = [c for c in selected_assets if c in prices_df.columns]
    if not available_assets:
        available_assets = list(prices_df.columns)

    start_year, end_year = selected_years

    df = prices_df[available_assets].copy()
    df = df.loc[
        (df.index >= f"{start_year}-01-01") &
        (df.index <= f"{end_year}-12-31")
    ]

    if df.empty:
        return df

    valid_ratio = 1.0 - df.isna().mean()
    keep_cols = valid_ratio[valid_ratio >= min_non_nan_ratio].index.tolist()

    if not keep_cols:
        return pd.DataFrame(index=df.index)

    df = df[keep_cols]

    df = df.dropna(how="all")
    df = df.ffill()
    df = df.dropna(how="any")

    return df


def _make_base_figure(title: str, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    full_title = title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>"

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=full_title,
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=18),
        ),
        height=700,
        margin=dict(l=60, r=20, t=100, b=150),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
        ),
        paper_bgcolor="#F6F7FB",
        plot_bgcolor="white",
        font=dict(size=13),
    )
    return fig


def _disabled_figure(title: str, message: str) -> go.Figure:
    fig = _make_base_figure(title)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15),
    )
    return fig


def _clamp_year_range(year_range: list[int], min_year: int, max_year: int) -> list[int]:
    start_year, end_year = year_range
    start_year = max(min_year, start_year)
    end_year = min(max_year, end_year)

    if start_year > end_year:
        return [min_year, max_year]

    return [start_year, end_year]


def _preset_year_range(preset_key: str, min_year: int, max_year: int) -> list[int]:
    # FIX: ranges now start one year before the shock so context is visible,
    # and extend one year after to show recovery.
    # Dot-com: crash peaked March 2000, trough Oct 2002 → show 1999–2003
    # GFC:     Lehman Sep 2008, trough Mar 2009 → show 2007–2010
    # COVID:   shock Feb–Mar 2020, recovery by end 2020 → show 2019–2021
    # Rate:    hike cycle started Mar 2022, bottomed Oct 2022 → show 2021–2023
    presets = {
        "dotcom":       [1999, 2003],
        "crisis_2008":  [2007, 2010],
        "covid_2020":   [2019, 2021],
        "rate_2022":    [2021, 2023],
    }

    if preset_key not in presets:
        return [min_year, max_year]

    return _clamp_year_range(presets[preset_key], min_year, max_year)


# ============================================================
# FORECAST HELPERS
# ============================================================

DATA_DIR = Path(r"D:\0Storage\bachelors\investment-dashboard\data")
FORECAST_ROOT = DATA_DIR / "forecast_prices"
FORECAST_DIRS = {
    "3mo": FORECAST_ROOT / "3mo",
    "1mo": FORECAST_ROOT / "1mo",
}
INFO_DIR = DATA_DIR / "asset_info"

VALID_FORECAST_MODES = {"none", "1mo", "3mo"}


def _forecast_path(ticker: str, forecast_mode: str) -> Path:
    if forecast_mode not in FORECAST_DIRS:
        raise ValueError("forecast_mode must be 'none', '1mo', or '3mo'")
    return FORECAST_DIRS[forecast_mode] / f"{ticker}.parquet"


def _load_forecast_close_series(ticker: str, forecast_mode: str) -> pd.Series:
    path = _forecast_path(ticker, forecast_mode)

    if not path.exists():
        return pd.Series(dtype="float64", name=ticker)

    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.Series(dtype="float64", name=ticker)

    if df.empty:
        return pd.Series(dtype="float64", name=ticker)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) not in {"", "nan", "None"}]).strip("_")
            for col in df.columns.to_list()
        ]

    if "Close" not in df.columns:
        return pd.Series(dtype="float64", name=ticker)

    close_obj = df.loc[:, "Close"]
    if isinstance(close_obj, pd.DataFrame):
        close_obj = close_obj.iloc[:, 0]

    close = pd.to_numeric(close_obj, errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype="float64", name=ticker)

    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.loc[~close.index.isna()]
    close.index = (
        close.index.tz_localize(None)
        if getattr(close.index, "tz", None) is not None
        else close.index
    )
    close.name = ticker
    return close.sort_index()


def _merge_prices_with_forecast(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    forecast_mode: str,
) -> pd.DataFrame:

    if forecast_mode not in {"1mo", "3mo"}:
        return prices_df.copy()

    merged = {}

    for asset in selected_assets:
        if asset not in prices_df.columns:
            continue

        hist = prices_df[asset].copy()
        hist.index = pd.to_datetime(hist.index, errors="coerce")
        hist = hist.loc[~hist.index.isna()]
        hist = hist.dropna()

        forecast = _load_forecast_close_series(asset, forecast_mode)

        if hist.empty and forecast.empty:
            continue

        s = pd.concat([hist, forecast], axis=0).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = asset
        merged[asset] = s

    if not merged:
        return prices_df.copy()

    combined = pd.concat(merged, axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index, errors="coerce")
    combined = combined.loc[~combined.index.isna()]
    return combined


def _effective_year_range_for_forecast(
    selected_years: list[int],
    prices_df: pd.DataFrame,
    forecast_mode: str,
) -> list[int]:
    """
    Extend the year range to include the forecast window.

    Both horizons (1mo = 21 days, 3mo = 63 days) end within the same
    calendar year as the last real data point, so we extend the end year
    by +1 only when the forecast window crosses a year boundary.
    """
    if forecast_mode not in {"1mo", "3mo"}:
        return selected_years

    from forecast import HORIZONS
    import pandas as _pd

    _, real_max_year = _year_bounds(prices_df)
    start_year = min(selected_years[0], real_max_year)

    # Find the actual last forecast date to determine whether we cross
    # into the next calendar year
    real_last = _real_last_date(prices_df)
    pred_len  = HORIZONS.get(forecast_mode, 63)
    if real_last is not None:
        last_forecast_date = real_last + _pd.offsets.BDay(pred_len)
        end_year = max(real_max_year, last_forecast_date.year)
    else:
        end_year = real_max_year + 1

    return [start_year, end_year]


def _real_last_date(prices_df: pd.DataFrame) -> pd.Timestamp | None:
    """Return the last date present in the real (non-forecast) prices DataFrame."""
    try:
        return pd.to_datetime(prices_df.index).max()
    except Exception:
        return None


# ============================================================
# ASSET INFO TABLE
# ============================================================

def _load_asset_info(ticker: str) -> dict:
    """Load metadata fields from asset_info parquet for one ticker."""
    path = INFO_DIR / f"{ticker}_info.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return row
    except Exception:
        return {}


def _fmt(val, pct: bool = False, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

RISK_FREE_RATE_ANNUAL = 0.04          # 4 % assumed annual risk-free rate
TRADING_DAYS = 252
BENCHMARK = "SPY"
BEAR_THRESHOLD = -0.01                # daily return < -1 % = "bear day"
CVAR_ALPHA = 0.05                     # 5 % tail


def _daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column pct_change without global dropna.
    NaNs are left in place so that a gap in one asset does NOT truncate
    the full history of all other assets.  Each metric function is
    responsible for dropping NaNs from its own slice.
    """
    return prices_df.pct_change()


def _annualized_return(ret: pd.Series) -> float:
    n = len(ret)
    if n < 2:
        return np.nan
    total = (1 + ret).prod()
    return float(total ** (TRADING_DAYS / n) - 1)


def _annualized_vol(ret: pd.Series) -> float:
    return float(ret.std() * np.sqrt(TRADING_DAYS))


def calc_sortino(ret: pd.Series) -> float:
    """
    Classic Sortino Ratio.

    Sortino = (Annualized Return − Rf) / Annualized Downside Deviation

    rf_daily  = (1 + rf_annual)^(1/252) − 1
    excess_t  = r_t − rf_daily
    downside_t = min(excess_t, 0)          # zero for days above hurdle
    Downside Deviation = sqrt(mean(downside_t²)) × sqrt(252)

    All days (not just negative days) enter the mean, which is the correct
    semi-deviation denominator (Sortino & van der Meer, 1991).
    """
    r = ret.dropna()
    n = len(r)
    if n < 2:
        return np.nan
    ann_ret  = _annualized_return(r)
    rf_daily = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0
    excess   = r - rf_daily
    downside = np.minimum(excess, 0.0)          # clip positives to zero
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))
    if downside_dev == 0:
        return np.nan
    return float((ann_ret - RISK_FREE_RATE_ANNUAL) / downside_dev)


def calc_upside_capture(ret: pd.Series, bench: pd.Series) -> float:
    """
    Upside Capture Ratio = mean(r_asset | r_SPY > 0) / mean(r_SPY | r_SPY > 0).
    Alignment and NaN removal are done per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if common2.empty:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]
    up_mask = b > 0
    if up_mask.sum() == 0:
        return np.nan
    bench_up_mean = b[up_mask].mean()
    if bench_up_mean == 0:
        return np.nan
    return float(r[up_mask].mean() / bench_up_mean)


def calc_downside_capture(ret: pd.Series, bench: pd.Series) -> float:
    """
    Downside Capture Ratio = mean(r_asset | r_SPY < 0) / mean(r_SPY | r_SPY < 0).
    Alignment and NaN removal are done per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if common2.empty:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]
    down_mask = b < 0
    if down_mask.sum() == 0:
        return np.nan
    bench_dn_mean = b[down_mask].mean()
    if bench_dn_mean == 0:
        return np.nan
    return float(r[down_mask].mean() / bench_dn_mean)


def calc_beta(ret: pd.Series, bench: pd.Series) -> float:
    """
    OLS Beta = Cov(r_asset, r_SPY) / Var(r_SPY).
    Asset and SPY are aligned by common dates; NaNs removed per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if len(common2) < 30:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]
    cov = np.cov(r, b)
    bench_var = cov[1, 1]
    if bench_var == 0:
        return np.nan
    return float(cov[0, 1] / bench_var)


def calc_bear_correlation(ret: pd.Series, bench: pd.Series) -> float:
    """
    Bear-Day Corr: Pearson correlation on days when r_SPY < -1%.
    Alignment and NaN removal per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if common2.empty:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]
    bear_mask = b < BEAR_THRESHOLD
    if bear_mask.sum() < 10:
        return np.nan
    return float(r[bear_mask].corr(b[bear_mask]))


def calc_portfolio_cvar_impact(
    ret: pd.Series,
    bench: pd.Series,
    alpha: float = CVAR_ALPHA,
) -> float:
    """
    50/50 Portfolio CVaR Impact.

    Measures the change in CVaR when blending this asset 50/50 with SPY:
        CVaR(0.5·r_asset + 0.5·r_SPY) − CVaR(r_SPY)

    Negative = asset reduces the tail risk of the benchmark portfolio (good).
    CVaR is the mean of daily returns in the left alpha-tail,
    annualised by linear scaling (×252).
    Alignment and NaN removal per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if len(common2) < 30:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]

    def _cvar(s: pd.Series) -> float:
        threshold = s.quantile(alpha)
        tail = s[s <= threshold]
        if tail.empty:
            return np.nan
        return float(tail.mean() * TRADING_DAYS)

    return float(_cvar(0.5 * r + 0.5 * b) - _cvar(b))


def calc_tail_dependence(ret: pd.Series, bench: pd.Series, alpha: float = CVAR_ALPHA) -> float:
    """
    Empirical Tail Dependence:
        P(r_asset ≤ q_asset,5% | r_SPY ≤ q_SPY,5%)

    Shows how often the asset is also in its worst 5% days when SPY is in
    its worst 5% days.  Lower = better for crisis diversification.
    Alignment and NaN removal per-pair.
    """
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if len(common2) < 30:
        return np.nan
    r, b = r.loc[common2], b.loc[common2]
    qr = r.quantile(alpha)
    qb = b.quantile(alpha)
    both_tail  = ((r <= qr) & (b <= qb)).sum()
    bench_tail = (b <= qb).sum()
    if bench_tail == 0:
        return np.nan
    return float(both_tail / bench_tail)


def calc_max_drawdown(ret: pd.Series) -> float:
    """
    Maximum Drawdown via wealth index starting at 1.0.

    W_t = ∏(1 + r_t)    (running product, W_0 = 1)
    M_t = max(W_0, ..., W_t)
    DD_t = W_t / M_t − 1
    """
    r = ret.dropna()
    if len(r) < 2:
        return np.nan
    wealth      = (1 + r).cumprod()
    running_max = wealth.cummax()
    drawdown    = wealth / running_max - 1
    return float(drawdown.min())


def calc_calmar(ret: pd.Series) -> float:
    """Calmar Ratio = Annualized Return / |Max Drawdown|."""
    r   = ret.dropna()
    ann_ret = _annualized_return(r)
    mdd     = calc_max_drawdown(r)
    if np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(ann_ret / abs(mdd))


def calc_correlation(ret: pd.Series, bench: pd.Series) -> float:
    """Plain Pearson correlation with benchmark. NaN removal per-pair."""
    common = ret.index.intersection(bench.index)
    if common.empty:
        return np.nan
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    if len(common2) < 10:
        return np.nan
    return float(r.loc[common2].corr(b.loc[common2]))


def compute_all_indicators(prices_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Compute all quantitative indicators for every ticker.

    _daily_returns() returns per-column pct_change WITHOUT global dropna so
    that a gap in one asset does not shorten other assets' history.
    Each metric function receives a per-asset or per-pair slice and applies
    its own dropna.

    Returns a DataFrame indexed by ticker.
    """
    all_rets  = _daily_returns(prices_df)
    bench_raw = all_rets[BENCHMARK] if BENCHMARK in all_rets.columns else None

    rows = []
    for ticker in tickers:
        if ticker not in all_rets.columns:
            continue

        r = all_rets[ticker]   # keep NaNs; each function handles its own dropna
        b = bench_raw          # same: may contain NaNs

        row = {"Ticker": ticker}

        # ── Profitability & Risk-Adjusted Return ─────────────
        row["Sortino"]     = calc_sortino(r)
        row["Calmar"]      = calc_calmar(r)

        # ── General Diversification & Market Risk ─────────────
        row["Beta"]                   = calc_beta(r, b)           if b is not None else np.nan
        row["Correlation"]            = calc_correlation(r, b)    if b is not None else np.nan
        row["Max Drawdown"]           = calc_max_drawdown(r)

        # ── Bear-Market Selection ─────────────────────────────
        row["Bear-Day Corr"]    = calc_bear_correlation(r, b)  if b is not None else np.nan
        row["Upside Capture"]   = calc_upside_capture(r, b)   if b is not None else np.nan
        row["Downside Capture"] = calc_downside_capture(r, b) if b is not None else np.nan

        # ── Crisis & Tail Diversification ────────────────────
        row["CVaR Impact"] = calc_portfolio_cvar_impact(r, b) if b is not None else np.nan
        row["Tail Dependence"]   = calc_tail_dependence(r, b)       if b is not None else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Ticker")


# ============================================================
# COLOUR CLASSIFICATION
# New rules: "std" = meets the academic/industry standard threshold (deep blue)
#            "good" = sufficiently good by expert assessment / better than SPY (light blue)
# ============================================================

def _classify_cell(
    col: str,
    val: float,
    spy_val: float | None = None,
) -> str | None:
    """
    Return 'std', 'good', or None.

    std  = meets the academic/industry standard threshold (deep blue)
    good = sufficiently good / better than SPY (light blue)

    ──────────────────────────────────────────────────────────────────
    INFO columns:
      5Y Avg Return   good  if > SPY value
      Beta 3Y         good  if < 0.5

    INDICATOR columns:
      Sortino         std   if > 1.0
                      good  if > SPY value
      Calmar          std   if > 1.0
                      good  if > SPY value

      Beta            std   if < 0    (negative beta = hedge)
                      good  if < 0.25
      Correlation     std   if < 0    (genuine diversifier)
                      good  if < 0.5
      Max Drawdown    good  if better (less negative) than SPY

      Bear-Day Corr   std   if < 0
                      good  if < 0.25
      Upside Capture  std   if > 1.0
                      good  if > 0.75
      Downside Capture
                      std   if < 0.0  (moves opposite to SPY on down-days)
                      good  if < 0.25

      50/50 Portfolio CVaR Impact
                      std   if < 0    (reduces tail risk)
                      good  if < 0.05
      Empirical Tail Dependence
                      good  if < 0.25
    ──────────────────────────────────────────────────────────────────
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    def _better_lower(ref: float | None) -> bool:
        return ref is not None and not np.isnan(ref) and val < ref

    def _better_higher(ref: float | None) -> bool:
        return ref is not None and not np.isnan(ref) and val > ref

    # ── INFO ─────────────────────────────────────────────────────
    if col == "5Y Avg Return":
        return "std" if _better_higher(spy_val) else None

    if col == "Beta 3Y":
        return "good" if val < 0.5 else None

    # ── Profitability & Risk-Adjusted Return ─────────────────────
    if col == "Sortino":
        if val > 1.0:
            return "std"
        return "good" if _better_higher(spy_val) else None

    if col == "Calmar":
        if val > 1.0:
            return "std"
        return "good" if _better_higher(spy_val) else None

    # ── General Diversification & Market Risk ────────────────────
    if col == "Beta":
        if val < 0:
            return "std"
        return "good" if val < 0.25 else None

    if col == "Correlation":
        if val < 0:
            return "std"
        return "good" if val < 0.5 else None

    if col == "Max Drawdown":
        return "good" if _better_higher(spy_val) else None   # closer to 0 = better

    # ── Bear-Market Selection ─────────────────────────────────────
    if col == "Bear-Day Corr":
        if val < 0:
            return "std"
        return "good" if val < 0.25 else None

    if col == "Upside Capture":
        if val > 1.0:
            return "std"
        return "good" if val > 0.75 else None

    if col == "Downside Capture":
        if val < 0.0:
            return "std"
        return "good" if val < 0.25 else None

    # ── Crisis & Tail Diversification ────────────────────────────
    if col == "CVaR Impact":
        return "good" if val > 0 else None

    if col == "Tail Dependence":
        return "good" if val < 0.25 else None

    return None


# ============================================================
# TABLE BUILDER
# ============================================================

def _build_info_table_rows(
    tickers: list[str],
    indicators_df: pd.DataFrame,
    info_map: dict[str, dict],
    forecast_mode: str = "none",
) -> list[html.Tr]:
    """
    Build all <tr> rows for the combined asset info + indicators table.

    Column groups (rendered as a top header row with colspans):
      YFinance / Dataset              : Ticker, Short Name, Category, 5Y Avg Return, Beta 3Y
      Bear-Market Selection           : Bear-Day Corr, Upside Capture, Downside Capture
      Crisis & Tail Diversification   : 50/50 Portfolio CVaR Impact, Empirical Tail Dependence
      Profitability & Risk-Adjusted   : Sortino, Calmar
    """

    # ── Column definitions ───────────────────────────────────────
    # INFO_COLS: (display_label, yfinance_key)
    INFO_COLS = [
        ("Short Name",    "shortName"),
        ("Category",      "category"),
        ("5Y Avg Return", "fiveYearAverageReturn"),
        ("Beta 3Y",       "beta3Year"),
    ]

    # INDICATOR_COLS in display order (matches group order below)
    INDICATOR_COLS = [
        # General Diversification & Market Risk
        "Beta",
        "Correlation",
        "Max Drawdown",
        # Bear-Market Selection
        "Bear-Day Corr",
        "Upside Capture",
        "Downside Capture",
        # Crisis & Tail Diversification
        "CVaR Impact",
        "Tail Dependence",
        # Profitability & Risk-Adjusted Return
        "Sortino",
        "Calmar",
    ]

    # Group header definitions: (label, colspan)
    # Ticker col is sticky and belongs to the first group.
    GROUPS = [
        ("YFinance",                   1 + len(INFO_COLS)),   # Ticker + 4 info cols
        ("General diversification", 3),
        ("Bear-Market",                 3),
        ("Crisis",         2),
        ("Profitability",  2),
    ]

    # ── Pre-extract SPY reference values ────────────────────────
    spy_indicator_vals: dict[str, float] = {}
    if BENCHMARK in indicators_df.index:
        for col in INDICATOR_COLS:
            if col in indicators_df.columns:
                v = indicators_df.loc[BENCHMARK, col]
                spy_indicator_vals[col] = float(v) if not pd.isna(v) else np.nan
            else:
                spy_indicator_vals[col] = np.nan
    else:
        spy_indicator_vals = {col: np.nan for col in INDICATOR_COLS}

    spy_info = info_map.get(BENCHMARK, {})
    spy_info_vals: dict[str, float] = {}
    for _, key in INFO_COLS:
        raw = spy_info.get(key, None)
        try:
            spy_info_vals[key] = float(raw) if raw is not None else np.nan
        except (TypeError, ValueError):
            spy_info_vals[key] = np.nan

    # ── Shared style atoms ───────────────────────────────────────
    th_group_style = {
        "padding": "6px 14px",
        "whiteSpace": "nowrap",
        "textAlign": "center",
        "fontSize": "11px",
        "fontWeight": "700",
        "letterSpacing": "0.04em",
        "textTransform": "uppercase",
        "color": "#CBD5E1",
        "backgroundColor": "#0F172A",
        "borderBottom": "1px solid #1E293B",
        "borderRight": "1px solid #334155",
        "position": "sticky",
        "top": "0",
        "zIndex": "2",
    }

    th_style = {
        "padding": "10px 14px",
        "whiteSpace": "nowrap",
        "textAlign": "center",
        "fontSize": "12px",
        "fontWeight": "700",
        "color": "#F1F5F9",
        "backgroundColor": "#1E293B",
        "borderBottom": "2px solid #334155",
        "position": "sticky",
        "top": "29px",   # pushed below the group header row
        "zIndex": "2",
    }

    th_ticker_group = dict(th_group_style, **{
        "textAlign": "left",
        "position": "sticky",
        "left": "0",
        "zIndex": "3",
    })

    th_ticker = dict(th_style, **{
        "textAlign": "left",
        "position": "sticky",
        "left": "0",
        "top": "29px",
        "zIndex": "3",
    })

    # ── Group header row ─────────────────────────────────────────
    group_cells = []
    for i, (label, span) in enumerate(GROUPS):
        style = dict(th_ticker_group if i == 0 else th_group_style)
        style["borderRight"] = "2px solid #334155"
        group_cells.append(html.Th(label, colSpan=span, style=style))

    group_row = html.Tr(group_cells)

    # ── Column header row ────────────────────────────────────────
    # Track which columns end a group so we can add a visual separator
    GROUP_LAST_COLS = {
        # last col index (0-based) in each group, counting from left:
        # group 0 ends at col index = len(INFO_COLS)  (Ticker + INFO_COLS)
        # group 1 ends at col index = len(INFO_COLS) + 4 - 1
        # etc.
        len(INFO_COLS): True,                       # end of YFinance group
        len(INFO_COLS) + 3: True,                   # end of General Diversification
        len(INFO_COLS) + 3 + 3: True,               # end of Bear-Market
        len(INFO_COLS) + 3 + 3 + 2: True,           # end of Crisis & Tail
    }

    header_cells = [html.Th("Ticker", style=th_ticker)]
    for idx, (label, _) in enumerate(INFO_COLS):
        s = dict(th_style)
        col_pos = idx + 1   # 0 = Ticker
        if col_pos in GROUP_LAST_COLS:
            s["borderRight"] = "2px solid #334155"
        header_cells.append(html.Th(label, style=s))

    for idx, col in enumerate(INDICATOR_COLS):
        s = dict(th_style)
        col_pos = len(INFO_COLS) + 1 + idx
        if col_pos in GROUP_LAST_COLS:
            s["borderRight"] = "2px solid #334155"
        header_cells.append(html.Th(col, style=s))

    header_row = html.Tr(header_cells)

    rows = [group_row, header_row]

    # ── Data rows ────────────────────────────────────────────────
    td_base = {
        "padding": "8px 14px",
        "whiteSpace": "nowrap",
        "textAlign": "center",
        "fontSize": "12px",
        "color": "#1E293B",
        "borderBottom": "1px solid #E2E8F0",
        "backgroundColor": "white",
    }

    # Group separator right-border positions (same col_pos logic)
    GROUP_SEP = set(GROUP_LAST_COLS.keys())

    for i, ticker in enumerate(tickers):
        row_bg = "white" if i % 2 == 0 else "#F8FAFC"
        info   = info_map.get(ticker, {})
        cells  = []

        # Ticker cell (sticky left)
        badge = {"1mo": " ᶠ¹", "3mo": " ᶠ³"}.get(forecast_mode, "")
        tip   = (f"Metrics include {forecast_mode} PatchTST forecast data"
                 if forecast_mode != "none" else ticker)
        cells.append(html.Td(
            f"{ticker}{badge}",
            title=tip,
            style={
                **td_base,
                "backgroundColor": "#1E293B",
                "color": "#93C5FD",
                "fontWeight": "700",
                "textAlign": "left",
                "position": "sticky",
                "left": "0",
                "zIndex": "1",
                "borderRight": "2px solid #334155",
            },
        ))

        # INFO columns
        for col_offset, (_, key) in enumerate(INFO_COLS):
            col_pos = col_offset + 1

            raw = info.get(key, None)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                display  = "—"
                cell_val = np.nan
            elif key == "fiveYearAverageReturn":
                try:
                    cell_val = float(raw)
                    display  = f"{cell_val * 100:.2f}%"
                except Exception:
                    display, cell_val = str(raw), np.nan
            elif key == "beta3Year":
                try:
                    cell_val = float(raw)
                    display  = f"{cell_val:.3f}"
                except Exception:
                    display, cell_val = str(raw), np.nan
            else:
                display, cell_val = str(raw), np.nan

            # Classify
            if key == "fiveYearAverageReturn":
                spy_ref = spy_info_vals.get(key, np.nan)
                tier = _classify_cell("5Y Avg Return", cell_val, spy_ref)
            elif key == "beta3Year":
                tier = _classify_cell("Beta 3Y", cell_val)
            else:
                tier = None

            sep = "2px solid #334155" if col_pos in GROUP_SEP else None

            if tier == "std":
                cell_style = {**td_base, "backgroundColor": BLUE_STD,
                              "color": TEXT_STD, "fontWeight": "700"}
            elif tier == "good":
                cell_style = {**td_base, "backgroundColor": BLUE_GOOD,
                              "color": TEXT_GOOD, "fontWeight": "600"}
            else:
                cell_style = {**td_base, "backgroundColor": row_bg}

            if sep:
                cell_style = {**cell_style, "borderRight": sep}

            cells.append(html.Td(display, style=cell_style))

        # INDICATOR columns
        for col_offset, col in enumerate(INDICATOR_COLS):
            col_pos = len(INFO_COLS) + 1 + col_offset

            val = (indicators_df.loc[ticker, col]
                   if col in indicators_df.columns and ticker in indicators_df.index
                   else np.nan)

            spy_ref = spy_indicator_vals.get(col, np.nan)
            tier = _classify_cell(col, val, spy_ref)

            # Format display
            is_nan = isinstance(val, float) and np.isnan(val)
            if col in ("Max Drawdown", "CVaR Impact"):
                display = _fmt(val, pct=True) if not is_nan else "—"
            elif col in ("Upside Capture", "Downside Capture", "Correlation",
                         "Bear-Day Corr", "Tail Dependence"):
                display = _fmt(val, decimals=3) if not is_nan else "—"
            else:
                display = _fmt(val, decimals=3) if not is_nan else "—"

            sep = "2px solid #334155" if col_pos in GROUP_SEP else None

            if tier == "std":
                cell_style = {**td_base, "backgroundColor": BLUE_STD,
                              "color": TEXT_STD, "fontWeight": "700"}
            elif tier == "good":
                cell_style = {**td_base, "backgroundColor": BLUE_GOOD,
                              "color": TEXT_GOOD, "fontWeight": "600"}
            else:
                cell_style = {**td_base, "backgroundColor": row_bg}

            if sep:
                cell_style = {**cell_style, "borderRight": sep}

            cells.append(html.Td(display, style=cell_style))

        rows.append(html.Tr(cells))

    return rows


def build_asset_table(
    tickers: list[str],
    prices_df: pd.DataFrame,
    forecast_mode: str = "none",
) -> html.Div:
    """
    Return the full scrollable metrics table.
    prices_df may be historical-only or historical+forecast merged.
    When a forecast is active the metrics reflect the predicted period.
    """
    indicators_df = compute_all_indicators(prices_df, tickers)
    info_map      = {t: _load_asset_info(t) for t in tickers}
    table_rows    = _build_info_table_rows(tickers, indicators_df, info_map,
                                           forecast_mode)

    legend_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "6px",
        "marginRight": "16px",
        "fontSize": "12px",
        "color": "#475569",
    }

    swatch_std = {
        "width": "14px",
        "height": "14px",
        "borderRadius": "3px",
        "backgroundColor": BLUE_STD,
        "display": "inline-block",
    }
    swatch_good = {
        "width": "14px",
        "height": "14px",
        "borderRadius": "3px",
        "backgroundColor": BLUE_GOOD,
        "display": "inline-block",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "Asset Overview & Diversification Metrics"
                        + {
                            "1mo": "  ·  Metrics include 1-month PatchTST forecast",
                            "3mo": "  ·  Metrics include 3-month PatchTST forecast",
                        }.get(forecast_mode, ""),
                        style={
                            "fontSize": "16px",
                            "fontWeight": "700",
                            "color": "#1E293B",
                            "marginRight": "24px",
                        },
                    ),
                    html.Span(
                        [html.Span(style=swatch_std), " Meets metric standard"],
                        style=legend_style,
                    ),
                    html.Span(
                        [html.Span(style=swatch_good), " Sufficiently good / better than SPY"],
                        style=legend_style,
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "flexWrap": "wrap",
                    "gap": "6px",
                },
            ),

            html.Div(
                html.Table(
                    table_rows,
                    style={
                        "borderCollapse": "collapse",
                        "width": "100%",
                        "minWidth": "max-content",
                    },
                ),
                style={
                    "overflowX": "auto",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
                    "border": "1px solid #E2E8F0",
                    "maxHeight": "420px",
                    "overflowY": "auto",
                    "position": "relative",
                },
            ),
        ],
        style={
            "backgroundColor": "white",
            "borderRadius": "14px",
            "padding": "18px 18px 14px 18px",
            "marginBottom": "18px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
        },
    )


# ============================================================
# CHARTS
# ============================================================

def build_indexed_price_figure(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
    real_cutoff_date: pd.Timestamp | None = None,
) -> go.Figure:
    df = _filter_selected_prices(prices_df, selected_assets, selected_years)
    start_year, end_year = selected_years
    chart_title = f"Indexed Asset Prices (Base = 100) ({start_year}–{end_year})"

    if df.empty:
        return _disabled_figure(
            chart_title,
            "No common time window exists for the selected assets and years.",
        )

    indexed = _normalize_growth(df)
    colors = get_palette(len(indexed.columns))
    fig = _make_base_figure(chart_title)

    for i, asset in enumerate(indexed.columns):
        s = indexed[asset].dropna()
        if s.empty:
            continue

        # Single continuous solid trace — historical and forecast as one line
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=asset,
            line=dict(color=colors[i], width=2),
            legendgroup=asset,
        ))

    # Subtle vertical marker at the real/forecast boundary
    if real_cutoff_date is not None and not indexed.empty:
        fig.add_vline(
            x=real_cutoff_date.timestamp() * 1000,
            line_dash="dot",
            line_color="rgba(0,0,0,0.25)",
            line_width=1.2,
            annotation_text="← Historical  |  Forecast →",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="rgba(0,0,0,0.45)",
        )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickangle=-30,
            nticks=10,
        ),
        yaxis=dict(
            title="Indexed Price (Base = 100)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickformat=".1f",
        ),
    )
    return fig


def build_pca_figure(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
) -> go.Figure:
    start_year, end_year = selected_years
    chart_title = f"PCA Return-Space Asset Map ({start_year}–{end_year})"

    df = _filter_selected_prices(prices_df, selected_assets, selected_years)
    if df.empty or len(df.columns) < 2:
        return _disabled_figure(
            chart_title,
            "At least two assets with common history are required for PCA.",
        )

    rets = df.pct_change().dropna()
    if rets.empty or len(rets.columns) < 2:
        return _disabled_figure(chart_title, "Not enough return data for PCA.")

    # PCA on assets (rows = assets, cols = daily return observations)
    X = rets.T
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(coords, index=X.index, columns=["PC1", "PC2"])

    fig = _make_base_figure(
        chart_title,
        subtitle=(
            f"Assets projected onto the two largest principal components of daily returns. "
            f"Proximity indicates similar return co-movement."
        ),
    )

    colors = get_palette(len(pca_df.index))
    fig.add_trace(
        go.Scatter(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            mode="markers+text",
            text=pca_df.index,
            textposition="top center",
            marker=dict(
                size=14,
                color=colors,
                line=dict(width=1, color="rgba(0,0,0,0.3)"),
            ),
            name="Assets",
            showlegend=False,
        )
    )

    fig.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_width=1, line_color="gray", opacity=0.5)

    ev1 = pca.explained_variance_ratio_[0]
    ev2 = pca.explained_variance_ratio_[1]

    fig.update_layout(
        xaxis=dict(
            title=f"PC1 — {ev1:.1%} of variance explained",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title=f"PC2 — {ev2:.1%} of variance explained",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
    )
    return fig


def build_rolling_correlation_figure(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
    benchmark: str = "SPY",
    window: int = 252,
) -> go.Figure:
    prices_df = _ensure_datetime_index(prices_df)
    start_year, end_year = selected_years
    chart_title = f"Rolling {window}D Correlation vs {benchmark} ({start_year}–{end_year})"

    if benchmark not in prices_df.columns:
        return _disabled_figure(
            chart_title,
            f"{benchmark} is not available in the loaded dataset.",
        )

    assets = [a for a in selected_assets if a in prices_df.columns and a != benchmark]
    if not assets:
        return _disabled_figure(
            chart_title,
            f"No comparable selected assets are available against {benchmark}.",
        )

    cols = [benchmark] + assets
    df = prices_df[cols].copy()
    df = df.loc[
        (df.index >= f"{start_year}-01-01") &
        (df.index <= f"{end_year}-12-31")
    ]
    df = df.dropna(how="any")

    if df.empty or len(df) <= window + 5:
        return _disabled_figure(
            chart_title,
            "Not enough overlapping data for rolling correlation.",
        )

    rets = df.pct_change().dropna()
    if rets.empty:
        return _disabled_figure(
            chart_title,
            "Not enough return observations for rolling correlation.",
        )

    corr_df = pd.DataFrame(
        {
            asset: rets[asset].rolling(window).corr(rets[benchmark])
            for asset in assets
        }
    ).dropna(how="all")

    if corr_df.empty:
        return _disabled_figure(
            chart_title,
            "Rolling correlation could not be calculated for the selected range.",
        )

    fig = _make_base_figure(chart_title)

    colors = get_palette(len(corr_df.columns))
    for i, col in enumerate(corr_df.columns):
        fig.add_trace(
            go.Scatter(
                x=corr_df.index,
                y=corr_df[col],
                mode="lines",
                name=col,
                line=dict(color=colors[i], width=2),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_width=1, line_color="gray")
    # Reference bands for interpretation
    fig.add_hrect(y0=-1, y1=0, fillcolor="rgba(37,99,235,0.04)", line_width=0)

    fig.update_layout(
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickangle=-30,
            nticks=10,
        ),
        yaxis=dict(
            title=f"Pearson Correlation with {benchmark}",
            range=[-1, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickformat=".2f",
        ),
    )
    return fig


def build_drawdown_smooth_figure(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
    span: int = 30,
) -> go.Figure:
    df = _filter_selected_prices(prices_df, selected_assets, selected_years)
    start_year, end_year = selected_years
    chart_title = f"Drawdown from Peak ({start_year}–{end_year}, EMA smoothing span={span}d)"

    if df.empty:
        return _disabled_figure(
            chart_title,
            "No common time window exists for the selected assets and years.",
        )

    growth = _normalize_growth(df)
    wealth = growth / 100.0
    wealth_smooth = wealth.ewm(span=span, adjust=False).mean()
    running_max = wealth_smooth.cummax()
    drawdown = wealth_smooth / running_max - 1.0

    colors = get_palette(len(drawdown.columns))
    fig = _make_base_figure(chart_title)

    for i, asset in enumerate(drawdown.columns):
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[asset],
                mode="lines",
                name=asset,
                line=dict(color=colors[i], width=2),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickangle=-30,
            nticks=10,
        ),
        yaxis=dict(
            title="Drawdown from Peak",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickformat=".0%",   # FIX: format as percentage (−10%, −20%, …)
        ),
    )
    return fig


def build_rolling_volatility_figure(
    prices_df: pd.DataFrame,
    selected_assets: list[str],
    selected_years: list[int],
    window: int = 63,
) -> go.Figure:
    start_year, end_year = selected_years
    chart_title = f"Rolling {window}D Annualised Volatility ({start_year}–{end_year})"

    df = _filter_selected_prices(prices_df, selected_assets, selected_years)
    if df.empty:
        return _disabled_figure(chart_title, "No data available.")

    returns = df.pct_change().dropna()
    rolling_vol = returns.rolling(window).std() * sqrt(TRADING_DAYS)

    fig = _make_base_figure(chart_title)

    colors = get_palette(len(rolling_vol.columns))

    for i, col in enumerate(rolling_vol.columns):
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[col],
                mode="lines",
                name=col,
                line=dict(color=colors[i], width=2),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickangle=-30,
            nticks=10,
        ),
        yaxis=dict(
            title="Annualised Volatility (σ)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            tickformat=".0%",   # FIX: display as percentage (e.g. 15%, 20%)
        ),
    )

    return fig


# ============================================================
# LAYOUT
# ============================================================

def build_layout(
    tickers: list[str],
    prices_df: pd.DataFrame,
    initial_figures: dict[str, go.Figure],
    initial_table: "html.Div | None" = None,
) -> html.Div:
    min_year, max_year = _year_bounds(prices_df)
    default_years = _default_year_range(prices_df)

    sidebar_style = {
        "position": "fixed",
        "top": "0",
        "left": "0",
        "bottom": "0",
        "width": "320px",
        "padding": "18px 16px",
        "backgroundColor": "#111827",
        "color": "white",
        "overflowY": "auto",
        "boxShadow": "2px 0 12px rgba(0,0,0,0.12)",
    }

    main_style = {
        "marginLeft": "340px",
        "padding": "18px 20px",
        "backgroundColor": "#F6F7FB",
        "minHeight": "100vh",
    }

    section_style = {
        "backgroundColor": "white",
        "borderRadius": "14px",
        "padding": "14px",
        "marginBottom": "14px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
    }

    btn_style = {
        "width": "100%",
        "marginTop": "8px",
        "padding": "8px",
        "borderRadius": "10px",
        "border": "1px solid #D1D5DB",
        "backgroundColor": "white",
        "cursor": "pointer",
        "textAlign": "left",
        "fontSize": "13px",
    }

    return html.Div(
        [
            # ── Sidebar ──────────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        "Decision Support System for Analysis of ETF Diversification Potential",
                        style={
                            "fontSize": "16px",
                            "fontWeight": "600",
                            "color": "#FFFFFF",
                            "opacity": "0.18",
                            "letterSpacing": "0.02em",
                            "lineHeight": "1.2",
                            "marginBottom": "14px",
                            "userSelect": "none",
                            "maxWidth": "100%",
                        },
                    ),

                    # Asset selector
                    html.Div(
                        [
                            html.Label("Assets", style={"fontWeight": "600", "color": "#111827"}),
                            dcc.Dropdown(
                                id="asset-selector",
                                options=[{"label": t, "value": t} for t in tickers],
                                value=tickers,
                                multi=True,
                                placeholder="Select assets",
                                style={"color": "#111827"},
                            ),
                        ],
                        style=section_style,
                    ),

                    # Year range + regime presets + forecast
                    html.Div(
                        [
                            html.Label("Year range", style={"fontWeight": "600", "color": "#111827"}),

                            dcc.RangeSlider(
                                id="year-range",
                                min=min_year,
                                max=max_year + 1,
                                step=1,
                                value=default_years,
                                marks=_build_year_marks(min_year, max_year + 1),
                                allowCross=False,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            # Quick regimes
                            html.Div(
                                [
                                    html.Label(
                                        "Quick regimes",
                                        style={"fontWeight": "600", "color": "#111827"},
                                    ),
                                    html.Button("2001 Dot-com crash",   id="btn-dotcom",         n_clicks=0, style=btn_style),
                                    html.Button("2008 Financial Crisis",        id="btn-crisis-2008",    n_clicks=0, style=btn_style),
                                    html.Button("2020 COVID shock",            id="btn-covid",          n_clicks=0, style=btn_style),
                                    html.Button("2022 Rate shock",             id="btn-rate-shock-2022", n_clicks=0, style=btn_style),
                                ],
                                style={"marginTop": "10px"},
                            ),

                            # Forecast horizon
                            html.Div(
                                [
                                    html.Label(
                                        "Forecast horizon",
                                        style={"fontWeight": "600", "color": "#111827"},
                                    ),
                                    html.Button("No forecast", id="btn-forecast-none", n_clicks=0, style=btn_style),
                                    html.Button("3 months",    id="btn-forecast-3mo",  n_clicks=0, style=btn_style),
                                    html.Button("1 month",     id="btn-forecast-1mo",  n_clicks=0, style=btn_style),
                                ],
                                style={"marginTop": "12px"},
                            ),

                            dcc.Store(id="forecast-mode", data="none"),
                        ],
                        style=section_style,
                    ),

                    # Visible charts
                    html.Div(
                        [
                            html.Label(
                                "Visible charts",
                                style={"fontWeight": "600", "color": "#111827"},
                            ),
                            dcc.Dropdown(
                                id="visible-charts",
                                options=[
                                    {"label": "Indexed prices",      "value": "indexed_price"},
                                    {"label": "PCA map",              "value": "pca"},
                                    {"label": "Rolling correlation",  "value": "rolling_corr"},
                                    {"label": "Drawdown",             "value": "drawdown"},
                                    {"label": "Rolling volatility",   "value": "rolling_vol"},
                                ],
                                value=[
                                    "indexed_price",
                                    "pca",
                                    "rolling_corr",
                                    "drawdown",
                                    "rolling_vol",
                                ],
                                multi=True,
                                placeholder="Select charts",
                                style={"color": "#111827"},
                            ),
                        ],
                        style=section_style,
                    ),

                    # No Refresh button — all controls are reactive

                ],
                style=sidebar_style,
            ),

            # ── Main content ─────────────────────────────────────────
            html.Div(
                id="charts-container",
                children=[
                    html.Div(
                        id="asset-table-container",
                        children=(initial_table
                              if initial_table is not None
                              else build_asset_table(tickers, prices_df)),
                    ),

                    dcc.Loading(type="circle", color="#2563EB", children=html.Div(
                        id="indexed-price-container",
                        children=dcc.Graph(
                            id="indexed-price-chart",
                            figure=initial_figures["indexed_price"],
                            config={"displayModeBar": False},
                        ),
                    )),

                    dcc.Loading(type="circle", color="#2563EB", children=html.Div(
                        id="pca-container",
                        children=dcc.Graph(
                            id="pca-chart",
                            figure=initial_figures["pca"],
                            config={"displayModeBar": False},
                        ),
                    )),

                    dcc.Loading(type="circle", color="#2563EB", children=html.Div(
                        id="rolling-container",
                        children=dcc.Graph(
                            id="rolling-corr-chart",
                            figure=initial_figures["rolling_corr"],
                            config={"displayModeBar": False},
                        ),
                    )),

                    dcc.Loading(type="circle", color="#2563EB", children=html.Div(
                        id="drawdown-container",
                        children=dcc.Graph(
                            id="drawdown-chart",
                            figure=initial_figures["drawdown"],
                            config={"displayModeBar": False},
                        ),
                    )),

                    dcc.Loading(type="circle", color="#2563EB", children=html.Div(
                        id="rolling-vol-container",
                        children=dcc.Graph(
                            id="rolling-vol-chart",
                            figure=initial_figures["rolling_vol"],
                            config={"displayModeBar": False},
                        ),
                    )),
                ],
                style=main_style,
            ),
        ]
    )


# ============================================================
# APP FACTORY
# ============================================================

def create_app(prices_df: pd.DataFrame, tickers: list[str]) -> Dash:
    prices_df = _ensure_datetime_index(prices_df)
    min_year, max_year = _year_bounds(prices_df)
    default_years = _default_year_range(prices_df)

    def _resolve_display_data(
        selected_assets: list[str],
        selected_years: list[int],
        forecast_mode: str,
    ) -> tuple:
        """Returns (display_prices_df, display_years, cutoff_date)."""
        if forecast_mode in {"1mo", "3mo"}:
            display_df    = _merge_prices_with_forecast(
                prices_df=prices_df,
                selected_assets=selected_assets,
                forecast_mode=forecast_mode,
            )
            display_years = _effective_year_range_for_forecast(
                selected_years=selected_years,
                prices_df=prices_df,
                forecast_mode=forecast_mode,
            )
            cutoff = _real_last_date(prices_df)
        else:
            display_df    = prices_df
            display_years = selected_years
            cutoff        = None
        return display_df, display_years, cutoff

    def _build_all_figures(
        selected_assets: list[str],
        selected_years: list[int],
        forecast_mode: str,
    ) -> dict[str, go.Figure]:
        """Central figure-building function used by initial load and all callbacks."""
        display_prices_df, display_years, cutoff = _resolve_display_data(
            selected_assets, selected_years, forecast_mode
        )

        return {
            "indexed_price": build_indexed_price_figure(
                prices_df=display_prices_df,
                selected_assets=selected_assets,
                selected_years=display_years,
                real_cutoff_date=cutoff,
            ),
            "pca": build_pca_figure(
                prices_df=display_prices_df,
                selected_assets=selected_assets,
                selected_years=display_years,
            ),
            "rolling_corr": build_rolling_correlation_figure(
                prices_df=display_prices_df,
                selected_assets=selected_assets,
                selected_years=display_years,
                benchmark="SPY",
                window=252,
            ),
            "drawdown": build_drawdown_smooth_figure(
                prices_df=display_prices_df,
                selected_assets=selected_assets,
                selected_years=display_years,
                span=30,
            ),
            "rolling_vol": build_rolling_volatility_figure(
                prices_df=display_prices_df,
                selected_assets=selected_assets,
                selected_years=display_years,
                window=63,
            ),
        }

    initial_figures = _build_all_figures(tickers, default_years, "none")
    initial_table   = build_asset_table(tickers, prices_df, forecast_mode="none")

    app = Dash(__name__)
    app.title = "ETF Diversification DSS"
    app.layout = build_layout(tickers, prices_df, initial_figures, initial_table)

    # ========================================================
    # QUICK REGIMES
    # ========================================================

    @app.callback(
        Output("year-range", "value", allow_duplicate=True),
        Input("btn-dotcom",          "n_clicks"),
        Input("btn-crisis-2008",     "n_clicks"),
        Input("btn-covid",           "n_clicks"),
        Input("btn-rate-shock-2022", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_regime_range(n_dotcom, n_2008, n_covid, n_rate):
        triggered = callback_context.triggered
        if not triggered:
            return default_years

        trigger_id = triggered[0]["prop_id"].split(".")[0]

        mapping = {
            "btn-dotcom":          "dotcom",
            "btn-crisis-2008":     "crisis_2008",
            "btn-covid":           "covid_2020",
            "btn-rate-shock-2022": "rate_2022",
        }

        key = mapping.get(trigger_id)
        if key:
            return _preset_year_range(key, min_year, max_year)

        return default_years

    # ========================================================
    # FORECAST BUTTONS → update store + year-range
    # ========================================================

    @app.callback(
        Output("forecast-mode", "data"),
        Output("year-range", "value"),
        Input("btn-forecast-none", "n_clicks"),
        Input("btn-forecast-3mo",  "n_clicks"),
        Input("btn-forecast-1mo",  "n_clicks"),
        State("year-range", "value"),
        prevent_initial_call=True,
    )
    def set_forecast_mode(n_none, n_1mo, n_3mo, current_years):
        triggered = callback_context.triggered
        if not triggered:
            return "none", current_years

        trigger_id = triggered[0]["prop_id"].split(".")[0]
        _, real_max_year = _year_bounds(prices_df)

        if not current_years or len(current_years) != 2:
            current_years = default_years

        start_year, end_year = _clamp_year_range(current_years, min_year, max_year + 1)
        start_year = min(start_year, real_max_year)

        # Compute the actual last forecast date to avoid hardcoding +1 year
        from forecast import HORIZONS as _H
        import pandas as _pd2
        real_last_dt = _real_last_date(prices_df)

        def _forecast_end_year(mode: str) -> int:
            pl = _H.get(mode, 63)
            if real_last_dt is not None:
                return max(real_max_year,
                           (real_last_dt + _pd2.offsets.BDay(pl)).year)
            return real_max_year + 1

        if trigger_id == "btn-forecast-none":
            return "none", [start_year, min(end_year, real_max_year)]

        if trigger_id == "btn-forecast-3mo":
            return "3mo", [start_year, _forecast_end_year("3mo")]

        if trigger_id == "btn-forecast-1mo":
            return "1mo", [start_year, _forecast_end_year("1mo")]

        return "none", current_years

    # ========================================================
    # CLAMP MANUAL SLIDER to real data bounds unless forecast active
    # ========================================================

    @app.callback(
        Output("year-range", "value", allow_duplicate=True),
        Input("year-range", "value"),
        State("forecast-mode", "data"),
        prevent_initial_call=True,
    )
    def clamp_year_range_by_mode(selected_years, forecast_mode):
        if not selected_years or len(selected_years) != 2:
            return default_years

        start_year, end_year = selected_years
        _, real_max_year = _year_bounds(prices_df)

        start_year = min(start_year, real_max_year)

        if forecast_mode in {"1mo", "3mo"}:
            return [start_year, real_max_year + 1]

        if end_year > real_max_year:
            return [start_year, real_max_year]

        return [start_year, end_year]

    # ========================================================
    # ASSET TABLE (refresh + asset-selector changes)
    # ========================================================

    @app.callback(
        Output("asset-table-container", "children"),
        Input("asset-selector", "value"),
        Input("forecast-mode",  "data"),
        Input("year-range",     "value"),
        prevent_initial_call=False,
    )
    def update_asset_table(selected_assets, forecast_mode, selected_years):
        active_tickers = selected_assets if selected_assets else tickers
        if forecast_mode not in VALID_FORECAST_MODES:
            forecast_mode = "none"
        sy = (selected_years if selected_years and len(selected_years) == 2
              else default_years)
        display_df, _, _ = _resolve_display_data(active_tickers, sy, forecast_mode)
        return build_asset_table(active_tickers, display_df, forecast_mode)

    # ========================================================
    # MAIN CHART REFRESH
    #
    # FIX: forecast-mode and year-range are now Inputs (not just States)
    # so selecting a forecast horizon or changing the range immediately
    # triggers a full redraw without requiring the Refresh button.
    # ========================================================

    @app.callback(
        Output("indexed-price-chart", "figure"),
        Output("pca-chart", "figure"),
        Output("rolling-corr-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("rolling-vol-chart", "figure"),
        Output("indexed-price-container", "style"),
        Output("pca-container", "style"),
        Output("rolling-container", "style"),
        Output("drawdown-container", "style"),
        Output("rolling-vol-container", "style"),
        Input("asset-selector",  "value"),
        Input("forecast-mode",   "data"),
        Input("year-range",      "value"),
        Input("visible-charts",  "value"),
        prevent_initial_call=False,
    )
    def update_all_charts(selected_assets, forecast_mode, selected_years, visible_charts):
        if not selected_assets:
            selected_assets = tickers

        if not selected_years or len(selected_years) != 2:
            selected_years = default_years

        if forecast_mode not in VALID_FORECAST_MODES:
            forecast_mode = "none"

        if visible_charts is None:
            visible_charts = []

        visible_charts = set(visible_charts)

        figs = _build_all_figures(selected_assets, selected_years, forecast_mode)

        def visible_style(chart_key: str) -> dict:
            return {"display": "block"} if chart_key in visible_charts else {"display": "none"}

        return (
            figs["indexed_price"],
            figs["pca"],
            figs["rolling_corr"],
            figs["drawdown"],
            figs["rolling_vol"],
            visible_style("indexed_price"),
            visible_style("pca"),
            visible_style("rolling_corr"),
            visible_style("drawdown"),
            visible_style("rolling_vol"),
        )

    return app


def run_dashboard(prices_df: pd.DataFrame, tickers: list[str], debug: bool = True) -> None:
    app = create_app(prices_df=prices_df, tickers=tickers)
    app.run(debug=debug, use_reloader=False)