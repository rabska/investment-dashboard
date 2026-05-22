from data_manager import DataManager
from dashboard_app import run_dashboard
from forecast import ForecastGenerator


TICKERS = """
    SPY QQQ IWM VXUS
    ARKK
    TLT IEF SHY TIP VTIP LQD HYG BND
    GLD SLV USO
    XLV XLU XLP VDC
    USMV SPLV
    SCHD DGRO VIG
    VNQ
    DBC PDBC
    DBMF KMLM
    """.split()


# ============================================================
# UPDATE MODE
# ============================================================
#
#   0 — frozen:      нічого не оновлювати і не перераховувати.
#                    Програма просто завантажує наявні дані та запускає дашборд.
#
#   1 — incremental: оновити дані з YFinance; перерахувати прогноз лише
#                    якщо з'явились нові ринкові дані або прогнозний кеш відсутній.
#                    (поведінка за замовчуванням / продакшн-режим)
#
#   2 — force:       оновити дані з YFinance; примусово перерахувати прогноз
#                    для ВСІХ тікерів незалежно від наявності нових даних.
#                    Використовується для тестування нових forecast-архітектур.
#
UPDATE_MODE: int = 0


def main():
    manager = DataManager()

    # ========================================================
    # GUARD: frozen mode — пропускаємо будь-яке оновлення
    # ========================================================

    if UPDATE_MODE == 0:
        print("UPDATE_MODE=0 (frozen): skipping data update and forecast rebuild.")

    # ========================================================
    # UPDATE REAL DATA  (режими 1 і 2)
    # ========================================================

    elif UPDATE_MODE in (1, 2):
        changed_tickers = manager.update_tickers(TICKERS)

        # ====================================================
        # FORECAST INVALIDATION / REBUILD
        # ====================================================

        if UPDATE_MODE == 2:
            # Форсований перерахунок: інвалідуємо все незалежно від змін
            print("\nUPDATE_MODE=2 (force): invalidating all forecasts for rebuild.")
            manager.invalidate_all_forecasts(TICKERS)

        elif changed_tickers:
            # Incremental: інвалідуємо лише якщо є нові дані
            print("\nReal data changed.")
            print("Invalidating and rebuilding forecasts...")
            manager.invalidate_all_forecasts(TICKERS)

        else:
            print("\nAll market data is already up to date.")
            print("Forecasts will only be generated if missing.")

        # ====================================================
        # FORECAST GENERATION
        # ====================================================

        forecast_generator = ForecastGenerator()

        try:
            forecast_generator.generate_forecasts(TICKERS)
            print("\nForecast pipeline completed.")
        except Exception as e:
            print("\nForecast pipeline failed:")
            print(e)

    else:
        raise ValueError(
            f"UPDATE_MODE must be 0, 1, or 2 — got {UPDATE_MODE!r}"
        )

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