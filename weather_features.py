"""
AgroMind — Модуль метеоданных для прогноза урожайности
Источник: NASA POWER API (исторические) + Open-Meteo (прогноз)

Использование:
    from weather_features import build_weather_features, get_forecast_weather_features
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─── Координаты по умолчанию (Кокшетау, Акмолинская область) ────────────────
DEFAULT_LAT = 53.28
DEFAULT_LON = 69.39

# ─── Путь к кешированным данным ──────────────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), "nasa_power_history_5y.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ОБРАБОТКА ИСТОРИЧЕСКИХ ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_nasa_history(path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Загружает исторические данные NASA POWER из CSV.
    Возвращает DataFrame с колонками:
        date, year, month, doy, temperature_2m, humidity_2m,
        wind_speed_2m, precipitation, solar_radiation, gdd
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"]   = df["date"].dt.day_of_year

    # Градусо-дни роста (GDD) — база 5°C (стандарт для зерновых)
    df["gdd"] = (df["temperature_2m"] - 5.0).clip(lower=0)

    return df


def update_nasa_history(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    years: int = 5,
    save_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """
    Скачивает свежие исторические данные NASA POWER и обновляет CSV.
    Вызывать раз в год или при смене координат поля.
    """
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    params = {
        "parameters": "T2M,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start_date.strftime("%Y%m%d"),
        "end":        end_date.strftime("%Y%m%d"),
        "format":     "JSON",
    }

    print(f"[weather] Скачиваем NASA POWER: {params['start']} — {params['end']}...")
    resp = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params=params, timeout=60
    )
    data = resp.json()

    if resp.status_code != 200:
        raise RuntimeError(f"NASA POWER error: {data}")

    parameters = data["properties"]["parameter"]
    dates = list(parameters["T2M"].keys())
    rows  = []

    for date in dates:
        t2m     = parameters["T2M"].get(date)
        rh2m    = parameters["RH2M"].get(date)
        ws2m    = parameters["WS2M"].get(date)
        prec    = parameters["PRECTOTCORR"].get(date)
        solar   = parameters["ALLSKY_SFC_SW_DWN"].get(date)

        if any(v is None or v <= -900 for v in [t2m, rh2m, ws2m, prec, solar]):
            continue

        rows.append({
            "date":            date,
            "temperature_2m":  t2m,
            "humidity_2m":     rh2m,
            "wind_speed_2m":   ws2m,
            "precipitation":   prec,
            "solar_radiation": solar,
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[weather] Сохранено {len(df)} записей → {save_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. АГРЕГАЦИЯ МЕТЕО-ФИЧ ПО ГОДУ (для обучения модели)
# ══════════════════════════════════════════════════════════════════════════════

def extract_year_features(df_daily: pd.DataFrame, year: int) -> dict:
    """
    Извлекает агрегированные метео-фичи за конкретный год.
    Делит сезон на 4 периода: весна, вегетация, уборка, год.

    Возвращает dict с ~20 числовыми фичами.
    """
    yr = df_daily[df_daily["year"] == year].copy()
    if yr.empty:
        return {}

    features = {"weather_year": year}

    # ── Апрель–Май: посев и всходы ──────────────────────────────
    spring = yr[yr["month"].isin([4, 5])]
    features["spring_precip_mm"]    = round(spring["precipitation"].sum(), 1)
    features["spring_temp_mean"]    = round(spring["temperature_2m"].mean(), 2)
    features["spring_gdd"]          = round(spring["gdd"].sum(), 1)
    features["spring_frost_days"]   = int((spring["temperature_2m"] < 0).sum())
    features["spring_dry_days"]     = int((spring["precipitation"] < 1).sum())
    features["spring_solar"]        = round(spring["solar_radiation"].mean(), 2)

    # ── Июнь–Июль: вегетация ────────────────────────────────────
    veg = yr[yr["month"].isin([6, 7])]
    features["veg_precip_mm"]       = round(veg["precipitation"].sum(), 1)
    features["veg_temp_mean"]       = round(veg["temperature_2m"].mean(), 2)
    features["veg_gdd"]             = round(veg["gdd"].sum(), 1)
    features["veg_heat_days"]       = int((veg["temperature_2m"] > 30).sum())  # стресс жары
    features["veg_solar"]           = round(veg["solar_radiation"].mean(), 2)
    features["veg_humidity_mean"]   = round(veg["humidity_2m"].mean(), 2)

    # ── Август: налив зерна и уборка ────────────────────────────
    aug = yr[yr["month"] == 8]
    features["aug_precip_mm"]       = round(aug["precipitation"].sum(), 1)
    features["aug_temp_mean"]       = round(aug["temperature_2m"].mean(), 2)
    features["aug_dry_days"]        = int((aug["precipitation"] < 1).sum())

    # ── Весь сезон (апрель–август) ──────────────────────────────
    season = yr[yr["month"].isin([4, 5, 6, 7, 8])]
    features["season_precip_mm"]    = round(season["precipitation"].sum(), 1)
    features["season_gdd_total"]    = round(season["gdd"].sum(), 1)
    features["season_temp_mean"]    = round(season["temperature_2m"].mean(), 2)

    # ── Индекс засухи (простой: осадки / испаряемость) ──────────
    # ET0 упрощённо = 0.0023 * (Tmean + 17.8) * sqrt(Tmax-Tmin) * Ra
    # Используем solar_radiation как proxy
    features["aridity_index"] = round(
        features["season_precip_mm"] / max(features["season_gdd_total"] * 0.1, 1), 3
    )

    return features


def build_weather_features(
    history_path: str = CACHE_PATH,
    years: list = None,
) -> pd.DataFrame:
    """
    Строит DataFrame с метео-фичами по годам — для объединения с датасетом модели.

    Возвращает DataFrame с колонками: year + ~20 метео-фич
    """
    df_daily = load_nasa_history(history_path)
    available_years = sorted(df_daily["year"].unique())

    if years:
        available_years = [y for y in available_years if y in years]

    rows = []
    for year in available_years:
        feats = extract_year_features(df_daily, year)
        if feats:
            rows.append(feats)

    df = pd.DataFrame(rows)
    # Переименовываем weather_year -> year для merge с основным датасетом
    if "weather_year" in df.columns:
        df = df.rename(columns={"weather_year": "year"})
    df["year"] = df["year"].astype("int64")
    print(f"[weather] Метео-фичи готовы: {len(df)} лет, {len(df.columns)-1} фич")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. ПРОГНОЗ НА БУДУЩИЙ СЕЗОН (Open-Meteo + исторический климат)
# ══════════════════════════════════════════════════════════════════════════════

def get_forecast_weather_features(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    target_year: int = None,
    history_path: str = CACHE_PATH,
) -> dict:
    """
    Получает метео-фичи для будущего сезона прогноза.

    Логика:
    1. Open-Meteo → текущий прогноз на 16 дней (если апрель-май)
    2. Исторический климатический средний за 5 лет → для остальных месяцев
    3. Объединяет в единый набор фич

    Возвращает dict с теми же фичами что и extract_year_features()
    """
    if target_year is None:
        target_year = datetime.now().year

    # ── Шаг 1: Исторический климатический средний ───────────────
    df_daily = load_nasa_history(history_path)
    hist_years = df_daily[df_daily["year"].isin(range(target_year - 5, target_year))]

    if hist_years.empty:
        hist_years = df_daily

    # Климатическая норма по месяцам
    climate_norm = hist_years.groupby("month").agg(
        precip_mean    = ("precipitation",   "mean"),
        temp_mean      = ("temperature_2m",  "mean"),
        gdd_mean       = ("gdd",             "mean"),
        solar_mean     = ("solar_radiation", "mean"),
        humidity_mean  = ("humidity_2m",     "mean"),
    ).reset_index()

    def climate_month(month, col):
        row = climate_norm[climate_norm["month"] == month]
        return float(row[col].values[0]) if not row.empty else 0.0

    # ── Шаг 2: Реальные данные 2026 из CSV (если есть) ──────────
    actual_data = df_daily[df_daily["year"] == target_year]
    actual_monthly = {}
    if not actual_data.empty:
        for m in actual_data["month"].unique():
            mdata = actual_data[actual_data["month"] == m]
            actual_monthly[int(m)] = {
                "precip": float(mdata["precipitation"].sum()),
                "temp":   float(mdata["temperature_2m"].mean()),
                "gdd":    float(mdata["gdd"].sum()),
                "days":   len(mdata),
            }
        print(f"[weather] Реальные данные {target_year}: {sorted(actual_monthly.keys())} месяцы")

    # ── Шаг 3: Open-Meteo прогноз на ближайшие 16 дней ──────────
    forecast_precip = {}
    forecast_temp   = {}
    current_month   = datetime.now().month

    if current_month in [4, 5, 6, 7]:
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude":    lat,
                "longitude":   lon,
                "daily":       "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 16,
                "timezone":    "auto",
            }, timeout=10)
            if resp.status_code == 200:
                daily = resp.json().get("daily", {})
                dates = daily.get("time", [])
                for i, d in enumerate(dates):
                    m = int(d[5:7])
                    tmax = daily["temperature_2m_max"][i] or 0
                    tmin = daily["temperature_2m_min"][i] or 0
                    prec = daily["precipitation_sum"][i] or 0
                    tmean = (tmax + tmin) / 2
                    if m not in forecast_precip:
                        forecast_precip[m] = []
                        forecast_temp[m]   = []
                    forecast_precip[m].append(prec)
                    forecast_temp[m].append(tmean)
                print(f"[weather] Open-Meteo прогноз загружен ({len(dates)} дней)")
        except Exception as e:
            print(f"[weather] Open-Meteo недоступен: {e}")

    # ── Шаг 4: Формируем фичи (реальные > прогноз > норма) ──────
    def month_precip(m):
        """Осадки за месяц. Реальные данные > прогноз > норма."""
        if m in actual_monthly:
            days = actual_monthly[m]["days"]
            raw  = actual_monthly[m]["precip"]
            if days >= 25:
                return raw   # полный месяц
            elif days >= 7:
                # Масштабируем на полный месяц
                return raw * (30 / days)
        if m in forecast_precip:
            return sum(forecast_precip[m])
        return climate_month(m, "precip_mean") * 30

    def month_temp(m):
        if m in actual_monthly and actual_monthly[m]["days"] >= 7:
            return actual_monthly[m]["temp"]
        elif m in forecast_temp:
            return np.mean(forecast_temp[m])
        return climate_month(m, "temp_mean")

    def month_gdd(m):
        if m in actual_monthly and actual_monthly[m]["days"] >= 7:
            days = actual_monthly[m]["days"]
            raw  = actual_monthly[m]["gdd"]
            if days >= 25:
                return raw
            return raw * (30 / days)
        t = month_temp(m)
        return max(0, (t - 5) * 30)

    features = {

        # Весна
        "spring_precip_mm": round(month_precip(4) + month_precip(5), 1),
        "spring_temp_mean": round((month_temp(4) + month_temp(5)) / 2, 2),
        "spring_gdd":       round(month_gdd(4) + month_gdd(5), 1),
        "spring_frost_days": round(climate_month(4, "temp_mean") < 2, 0) * 5,
        "spring_dry_days":  round((1 - min(month_precip(4) / 40, 1)) * 20),
        "spring_solar":     round(climate_month(5, "solar_mean"), 2),

        # Вегетация
        "veg_precip_mm":    round(month_precip(6) + month_precip(7), 1),
        "veg_temp_mean":    round((month_temp(6) + month_temp(7)) / 2, 2),
        "veg_gdd":          round(month_gdd(6) + month_gdd(7), 1),
        "veg_heat_days":    round(max(0, month_temp(7) - 25) * 3),
        "veg_solar":        round(climate_month(7, "solar_mean"), 2),
        "veg_humidity_mean":round(climate_month(6, "humidity_mean"), 2),

        # Август
        "aug_precip_mm":    round(month_precip(8), 1),
        "aug_temp_mean":    round(month_temp(8), 2),
        "aug_dry_days":     round((1 - min(month_precip(8) / 30, 1)) * 15),

        # Сезон
        "season_precip_mm": round(sum(month_precip(m) for m in [4,5,6,7,8]), 1),
        "season_gdd_total": round(sum(month_gdd(m) for m in [4,5,6,7,8]), 1),
        "season_temp_mean": round(np.mean([month_temp(m) for m in [4,5,6,7,8]]), 2),
    }
    features["aridity_index"] = round(
        features["season_precip_mm"] / max(features["season_gdd_total"] * 0.1, 1), 3
    )

    has_actual = len([m for m in actual_monthly if actual_monthly[m]["days"] >= 25]) > 0
    if has_actual and forecast_precip:
        source = "nasa_actual+open_meteo"
    elif has_actual:
        source = "nasa_actual+climate_norm"
    elif forecast_precip:
        source = "open_meteo+climate_norm"
    else:
        source = "climate_norm"
    features["weather_source"] = source
    print(f"[weather] Прогноз метео готов (источник: {source})")
    return features


# ══════════════════════════════════════════════════════════════════════════════
# 4. ИНТЕГРАЦИЯ С train_all_years.py
# ══════════════════════════════════════════════════════════════════════════════

def add_weather_to_dataset(
    df: pd.DataFrame,
    history_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """
    Добавляет метео-фичи к датасету модели.
    Вызывать в build_dataset() перед обучением.

    df должен содержать колонку 'year'.
    """
    years = df["year"].unique().tolist()
    weather_df = build_weather_features(history_path, years=years)

    if weather_df.empty:
        print("[weather] Нет метео-данных для объединения")
        return df

    df = df.merge(weather_df, on="year", how="left")
    filled = df["spring_precip_mm"].notna().sum()
    print(f"[weather] Метео добавлено для {filled}/{len(df)} записей")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. ТЕСТ
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Тест: исторические метео-фичи по годам")
    print("=" * 55)

    df_weather = build_weather_features()
    print()
    print(df_weather[["weather_year", "spring_precip_mm", "veg_precip_mm",
                       "aug_precip_mm", "season_gdd_total", "aridity_index"]].to_string(index=False))

    print()
    print("=" * 55)
    print("  Тест: прогноз на 2026")
    print("=" * 55)
    forecast = get_forecast_weather_features(target_year=2026)
    for k, v in forecast.items():
        print(f"  {k}: {v}")
