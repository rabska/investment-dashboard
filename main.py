from data_manager import DataManager
from dashboard_app import run_dashboard
from forecast import ForecastGenerator


TICKERS = """
SPY VXUS
TLT IEF SHY TIP VTIP LQD BND
GLD SLV USO
XLV XLU XLP VDC
USMV SPLV
SCHD DGRO VIG
VNQ
DBC PDBC
DBMF KMLM
""".split()


def main():
    manager = DataManager()

    # ========================================================
    # UPDATE REAL DATA
    # ========================================================

    changed_tickers = manager.update_tickers(TICKERS)

    # ========================================================
    # FORECAST INVALIDATION / REBUILD
    # ========================================================

    if changed_tickers:
        print("\nReal data changed.")
        print("Invalidating and rebuilding forecasts...")

        manager.invalidate_all_forecasts(TICKERS)

    else:
        print("\nAll market data is already up to date.")
        print("Forecasts will only be generated if missing.")

    # ========================================================
    # FORECAST GENERATION
    # ========================================================

    forecast_generator = ForecastGenerator()

    try:
        forecast_generator.generate_forecasts(TICKERS)
        print("\nForecast pipeline completed.")
    except Exception as e:
        print("\nForecast pipeline failed:")
        print(e)

    # ========================================================
    # LOAD DASHBOARD DATA
    # ========================================================

    prices_df = manager.load_prices(TICKERS)

    if prices_df.empty:
        raise RuntimeError(
            "Prices dataframe is empty. Check local parquet files."
        )

    print("\nLaunching dashboard...")

    # ========================================================
    # RUN DASHBOARD
    # ========================================================

    run_dashboard(
        prices_df=prices_df,
        tickers=TICKERS,
        debug=True,
    )


if __name__ == "__main__":
    main()