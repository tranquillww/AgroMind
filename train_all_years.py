"""
AgroMind — Обучение модели на всех годах
Запускать: python train_all_years.py
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from weather_features import add_weather_to_dataset, get_forecast_weather_features
from sklearn.model_selection import cross_val_score, KFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Пути к данным ───────────────────────────────────────
# Укажи папку где лежат все JSON файлы от данные с техники
DATA_DIR = os.environ.get("_DATA_DIR", "data/jd")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# Маппинг файлов по годам
HARVEST_FILES = {
    2021: "Урожайность(2021).json",
    2022: "Урожайность(2022).json",
    2023: "Урожайность(2023).json",
    2024: "Урожайность(2024).json",
    2025: "Урожайность(2025).json",
}
SEEDING_FILES = {
    2022: "Посев(seeding 2022).json",
    2023: "Посев(seeding 2023).json",
    2024: "Посев(seeding 2024).json",
    2025: "Посев(seeding 2025).json",
}
TILLAGE_FILES = {
    2021: "Обратботка почвы(tillage 2021).json",
    2023: "Обратботка почвы(tillage 2023).json",
    2024: "Обратботка почвы(tillage 2024).json",
    2025: "Обратботка почвы(tillage 2025).json",
}

# Культуры-исключения (нет урожайности в т/га зерна)
JUNK_CROPS = {"Не определена", "Без культуры"}

# Агрономический коэффициент предшественника
# Источник: агрономические рекомендации для Северного Казахстана
# 1.0 = нейтральный, >1 = улучшает урожай следующей культуры
PREDECESSOR_BENEFIT = {
    "PEAS_FIELD":    1.20,   # бобовые — лучший предшественник, фиксируют азот
    "PEA_TRAPPER":   1.20,
    "CANOLA":        1.15,   # рапс — хороший предшественник для злаков
    "RAPE_SEED":     1.15,
    "RAPE_SEED_E_OIL": 1.15,
    "FLAX":          1.10,   # лён — хороший предшественник
    "SUNFLOWER_OIL": 1.05,
    "SUNFLOWER_E_OIL": 1.05,
    "CORN_SILAGE":   1.05,
    "CORN_WET":      1.05,
    "BARLEY":        1.00,   # нейтральный
    "OATS":          1.00,
    "WHEAT_WHITE":   0.93,   # пшеница после пшеницы — хуже
    "WHEAT_DURUM":   0.93,
    "WHEAT_SFT_RD_WTR": 0.93,
    "WHEAT_HRD_RD_WTR": 0.93,
    "WHEAT_EURO_WTR":   0.93,
    "GRASS_FORAGE":  1.08,   # многолетние травы — хороший предшественник
}


# ─── Парсеры ─────────────────────────────────────────────

def _rows(data):
    return data.get("rows") or data.get("data", {}).get("rows", [])


def load_harvest(path, year):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for r in _rows(data):
        field_id = r.get("field", {}).get("id")
        if not field_id:
            continue
        area = r.get("area", {}).get("value", 0)
        if (area or 0) < 0.5:
            continue
        ctx = r.get("work", {}).get("context", {})
        records.append({
            "field_id":          field_id,
            "year":              year,
            "harvest_date":      pd.to_datetime(ctx.get("startDate") or r.get("lastTimestamp"), utc=True, errors="coerce"),
            "crop_type":         r.get("crop", {}).get("label", ""),
            "crop_token":        r.get("crop", {}).get("context", {}).get("token", ""),
            "harvest_area_ha":   area,
            "yield_t_ha":        r.get("averageWetWeight", {}).get("value"),
            # Дополнительные поля — качество уборки
            "harvest_moisture":  r.get("averageMoisture", {}).get("value"),      # влажность зерна %
            "dry_matter_pct":    r.get("dryMatter", {}).get("value"),             # сухое вещество %
            "harvest_speed":     r.get("speed", {}).get("value"),                 # скорость км/ч
            "harvest_prod":      r.get("productivity", {}).get("value"),          # производительность га/ч
            "harvest_fuel_l_ha": r.get("fuelProductivity", {}).get("value"),      # расход топлива л/га
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Фильтр аномальных урожайностей — агрономические максимумы для региона
    CROP_YIELD_MAX = {
        "WHEAT_DURUM": 4.5, "WHEAT_WHITE": 4.5, "WHEAT_SFT_RD_WTR": 4.5,
        "WHEAT_HRD_RD_WTR": 4.5, "WHEAT_EURO_WTR": 5.0,
        "BARLEY": 5.5, "BARLEY_EURO_SPR": 5.5,
        "OATS": 5.5, "OATS_EURO": 5.5,
        "FLAX": 2.5, "CANOLA": 3.5, "RAPE_SEED": 3.5, "RAPE_SEED_E_OIL": 3.5,
        "PEAS_FIELD": 3.5, "PEA_TRAPPER": 3.5,
        "SUNFLOWER_OIL": 4.0, "SUNFLOWER_E_OIL": 4.5,
        "CORN_WET": 12.0, "CORN_SILAGE": 55.0,
        "GRASS_FORAGE": 30.0, "MILLET": 2.5, "POTATOES_FOR_RETAIL": 50.0,
    }
    # Минимальная площадь для надёжной записи (маленькие площади = мусорные проходы)
    CROP_MIN_AREA = {
        "CORN_SILAGE": 5.0, "CORN_WET": 3.0,
        "SUNFLOWER_E_OIL": 3.0, "SUNFLOWER_OIL": 3.0,
    }
    def is_valid(row):
        token = row["crop_token"]
        yld = row["yield_t_ha"]
        area = row["harvest_area_ha"]
        if yld is None or yld <= 0:
            return False
        max_y = CROP_YIELD_MAX.get(token, 8.0)
        min_a = CROP_MIN_AREA.get(token, 1.0)
        return yld <= max_y and area >= min_a
    df = df[df.apply(is_valid, axis=1)]

    agg = df.groupby(["field_id", "year", "crop_type", "crop_token"]).agg(
        harvest_date     = ("harvest_date",      "min"),
        harvest_area_ha  = ("harvest_area_ha",   "sum"),
        yield_t_ha       = ("yield_t_ha",        "mean"),
        harvest_moisture = ("harvest_moisture",  "mean"),
        dry_matter_pct   = ("dry_matter_pct",    "mean"),
        harvest_speed    = ("harvest_speed",     "mean"),
        harvest_prod     = ("harvest_prod",      "mean"),
        harvest_fuel_l_ha= ("harvest_fuel_l_ha", "mean"),
        harvest_passes   = ("harvest_area_ha",   "count"),  # кол-во проходов уборщика
    ).reset_index()
    return agg


def load_seeding(path, year):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for r in _rows(data):
        field_id = r.get("field", {}).get("id")
        if not field_id:
            continue
        ctx = r.get("work", {}).get("context", {})
        records.append({
            "field_id":        field_id,
            "year":            year,
            "seeding_date":    pd.to_datetime(ctx.get("startDate") or r.get("lastTimestamp"), utc=True, errors="coerce"),
            "crop_token":      ctx.get("cropToken", ""),
            "variety":         r.get("variety", {}).get("label", ""),
            "seeding_area_ha": r.get("area", {}).get("value"),
            "seeding_speed":   r.get("speed", {}).get("value"),
            "seed_rate_kg_ha": r.get("targetRate", {}).get("value"),   # норма высева кг/га
            "seeding_fuel_l_ha": r.get("fuelProductivity", {}).get("value"),  # расход топлива
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.groupby(["field_id", "year", "crop_token"]).agg(
        seeding_date      = ("seeding_date",      "min"),
        variety           = ("variety",           "first"),
        seeding_area_ha   = ("seeding_area_ha",   "sum"),
        seeding_speed     = ("seeding_speed",     "mean"),
        seed_rate_kg_ha   = ("seed_rate_kg_ha",   "mean"),
        seeding_fuel_l_ha = ("seeding_fuel_l_ha", "mean"),
    ).reset_index()


def load_tillage(path, year):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for r in _rows(data):
        field_id = r.get("field", {}).get("id")
        if not field_id:
            continue
        ctx = r.get("work", {}).get("context", {})
        ts = ctx.get("startDate") or r.get("lastTimestamp", "")
        till_date = pd.to_datetime(ts, utc=True, errors="coerce")
        month = till_date.month if pd.notna(till_date) else 0
        records.append({
            "field_id":       field_id,
            "year":           year,
            "tillage_type":   ctx.get("tillageType", ""),
            "tillage_area":   r.get("area", {}).get("value", 0),
            "depth_cm":       r.get("targetDepth", {}).get("value"),
            "fuel_l_ha":      r.get("fuelProductivity", {}).get("value"),
            "till_month":     month,
            "is_autumn_till": int(month in [9, 10, 11]),   # осенняя зяблевая обработка
            "speed_kmh":      r.get("speed", {}).get("value"),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df[df["tillage_area"].fillna(0) > 0]
    return df.groupby(["field_id", "year"]).agg(
        tillage_type       = ("tillage_type",   "first"),
        tillage_operations = ("tillage_area",   "count"),
        avg_depth_cm       = ("depth_cm",       "mean"),
        tillage_fuel_l_ha  = ("fuel_l_ha",      "mean"),
        autumn_tillage     = ("is_autumn_till",  "max"),   # 1 если была осенняя обработка
        tillage_speed      = ("speed_kmh",       "mean"),
    ).reset_index()


# ─── Сборка датасета ──────────────────────────────────────

def build_dataset(data_dir=DATA_DIR):
    harvests, seedings, tillages = [], [], []

    for year, fname in HARVEST_FILES.items():
        path = Path(data_dir) / fname
        if path.exists():
            df = load_harvest(str(path), year)
            if not df.empty:
                harvests.append(df)
                print(f"  harvest {year}: {len(df)} записей")

    for year, fname in SEEDING_FILES.items():
        path = Path(data_dir) / fname
        if path.exists():
            df = load_seeding(str(path), year)
            if not df.empty:
                seedings.append(df)
                print(f"  seeding {year}: {len(df)} записей")

    for year, fname in TILLAGE_FILES.items():
        path = Path(data_dir) / fname
        if path.exists():
            df = load_tillage(str(path), year)
            if not df.empty:
                tillages.append(df)
                print(f"  tillage {year}: {len(df)} записей")

    if not harvests:
        raise ValueError("Нет файлов урожайности!")

    # Строим историю культур по полям для вычисления предшественника
    from collections import defaultdict
    field_crops = defaultdict(dict)
    for df_h in harvests:
        for _, row in df_h.iterrows():
            if row.get("crop_token") and row["crop_token"] not in ("NONE", "UNSPECIFIED_CROP", ""):
                field_crops[row["field_id"]][row["year"]] = row["crop_token"]

    harvest_df = pd.concat(harvests, ignore_index=True)
    seeding_df = pd.concat(seedings, ignore_index=True) if seedings else pd.DataFrame()
    tillage_df = pd.concat(tillages, ignore_index=True) if tillages else pd.DataFrame()

    # Фильтр мусорных культур
    harvest_df = harvest_df[~harvest_df["crop_type"].isin(JUNK_CROPS)]

    # Объединяем
    df = harvest_df.copy()

    if not seeding_df.empty:
        df = df.merge(
            seeding_df[["field_id", "year", "crop_token", "seeding_date",
                         "variety", "seeding_area_ha", "seeding_speed",
                         "seed_rate_kg_ha", "seeding_fuel_l_ha"]],
            on=["field_id", "year", "crop_token"], how="left"
        )

    if not tillage_df.empty:
        df = df.merge(
            tillage_df[["field_id", "year", "tillage_type", "tillage_operations",
                         "avg_depth_cm", "tillage_fuel_l_ha", "autumn_tillage",
                         "tillage_speed"]],
            on=["field_id", "year"], how="left"
        )

    # ─── Предшественник культуры ───
    # Строим lookup: (field_id, year) -> crop_token предыдущего года
    predecessor_map = {}
    for field_id, years_dict in field_crops.items():
        sorted_years = sorted(years_dict.keys())
        for i in range(1, len(sorted_years)):
            year = sorted_years[i]
            prev_year = sorted_years[i - 1]
            if year - prev_year <= 2:  # учитываем пар (1-2 года)
                predecessor_map[(field_id, year)] = years_dict[prev_year]

    df["predecessor_token"] = df.apply(
        lambda r: predecessor_map.get((r["field_id"], r["year"]), ""), axis=1
    )
    df["predecessor_benefit"] = df["predecessor_token"].map(
        lambda t: PREDECESSOR_BENEFIT.get(t, 1.0)
    )
    # Бинарный флаг: был ли бобовый предшественник
    df["prev_legume"] = (df["predecessor_token"].isin(
        {"PEAS_FIELD", "PEA_TRAPPER"}
    )).astype(int)
    # Бинарный флаг: пшеница после пшеницы (плохо)
    wheat_tokens = {"WHEAT_WHITE", "WHEAT_DURUM", "WHEAT_SFT_RD_WTR",
                    "WHEAT_HRD_RD_WTR", "WHEAT_EURO_WTR", "WHEAT_HRD_RD_WTR"}
    df["wheat_after_wheat"] = (
        df["predecessor_token"].isin(wheat_tokens) &
        df["crop_token"].isin(wheat_tokens)
    ).astype(int)

    print(f"  Предшественник известен: {df['predecessor_token'].ne('').sum()} / {len(df)} записей")

    # ─── Метеоданные NASA POWER ───
    try:
        df = add_weather_to_dataset(df)
    except Exception as e:
        print(f"  [weather] Пропуск метео: {e}")

    # ─── Площадь посева vs уборки ───
    if "seeding_area_ha" in df.columns:
        df["area_ratio"] = (
            df["seeding_area_ha"] / df["harvest_area_ha"].replace(0, np.nan)
        ).clip(0.5, 1.5)  # потери площади: <1 = часть не убрана, >1 = ошибка данных

    # ─── Влажность зерна — индикатор качества сезона ───
    # Высокая влажность = дождливый сезон или ранняя уборка
    if "harvest_moisture" in df.columns:
        df["moisture_stress"] = (df["harvest_moisture"].fillna(14) - 14).clip(-5, 20)
        # Отклонение от стандарта 14% — выше = влажный год, ниже = засушливый

    # ─── Интенсивность посева ───
    if "seed_rate_kg_ha" in df.columns:
        # Заполняем типичными нормами по культуре где нет данных
        TYPICAL_SEED_RATES = {
            "WHEAT_DURUM": 120, "WHEAT_WHITE": 120, "WHEAT_SFT_RD_WTR": 120,
            "WHEAT_HRD_RD_WTR": 120, "WHEAT_EURO_WTR": 120,
            "BARLEY": 150, "BARLEY_EURO_SPR": 150,
            "OATS": 170, "OATS_EURO": 170,
            "FLAX": 55, "CANOLA": 8, "RAPE_SEED": 8,
            "PEAS_FIELD": 180, "PEA_TRAPPER": 180,
            "SUNFLOWER_OIL": 6, "SUNFLOWER_E_OIL": 6,
            "CORN_WET": 20, "CORN_SILAGE": 20,
        }
        df["seed_rate_filled"] = df.apply(
            lambda r: r["seed_rate_kg_ha"] if pd.notna(r.get("seed_rate_kg_ha")) and r.get("seed_rate_kg_ha", 0) > 0
            else TYPICAL_SEED_RATES.get(r["crop_token"], 120),
            axis=1
        )

    # Производные фичи
    df["harvest_doy"]   = df["harvest_date"].dt.day_of_year
    df["harvest_month"] = df["harvest_date"].dt.month
    if "seeding_date" in df.columns:
        df["seeding_doy"]   = df["seeding_date"].dt.day_of_year
        df["seeding_month"] = df["seeding_date"].dt.month
        df["growing_days"]  = (df["harvest_date"] - df["seeding_date"]).dt.days

    return df


# ─── Модель ───────────────────────────────────────────────

class YieldForecasterV2:
    """
    Улучшенная модель прогноза урожайности — все годы.
    """

    EXCLUDE_FROM_FEATURES = {
        "yield_t_ha", "field_id", "crop_type", "crop_token",
        "harvest_date", "seeding_date", "harvest_area_ha",
        "variety", "tillage_type", "predecessor_token",
        # Post-harvest данные — УТЕЧКА (недоступны до уборки)
        "harvest_moisture", "dry_matter_pct",
        "harvest_speed", "harvest_prod", "harvest_fuel_l_ha",
        "moisture_stress",
    }

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.crop_stats = {}
        self.trained_at = None
        self.cv_results = {}
        self.dataset_info = {}

    def _encode(self, df, fit=False):
        df = df.copy()
        for col in ["crop_token", "crop_type", "tillage_type", "variety"]:
            if col not in df.columns:
                continue
            vals = df[col].fillna("—").astype(str)
            if fit:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(vals)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[f"{col}_enc"] = vals.map(
                        lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
                    )
        return df

    def fit(self, df: pd.DataFrame):
        print(f"\n{'='*55}")
        print(f"  Обучение YieldForecaster v2")
        print(f"  Записей: {len(df)} | Полей: {df['field_id'].nunique()} | Лет: {sorted(df['year'].unique())}")
        print(f"{'='*55}")

        # Статистика по культурам — используем медиану (устойчива к выбросам)
        raw_stats = df.groupby("crop_token")["yield_t_ha"].agg(
            ["mean", "median", "std", "count"]
        )

        # Региональные нормы как fallback (БНС РК, Акмолинская область)
        REGIONAL_NORMS = {
            "WHEAT_DURUM":        1.50, "WHEAT_WHITE":       1.80,
            "WHEAT_SFT_RD_WTR":   1.80, "WHEAT_HRD_RD_WTR":  1.70,
            "WHEAT_EURO_WTR":     2.00, "BARLEY":            2.10,
            "BARLEY_EURO_SPR":    2.10, "OATS":              2.20,
            "OATS_EURO":          2.20, "FLAX":              0.95,
            "CANOLA":             1.80, "RAPE_SEED":         1.80,
            "RAPE_SEED_E_OIL":    1.80, "PEAS_FIELD":        1.40,
            "PEA_TRAPPER":        1.40, "SUNFLOWER_OIL":     1.50,
            "SUNFLOWER_E_OIL":    1.70, "CORN_WET":          5.00,
            "CORN_SILAGE":       20.00, "GRASS_FORAGE":      15.0,
            "MILLET":             0.80, "POTATOES_FOR_RETAIL":25.0,
        }

        # Считаем кол-во уникальных лет по культуре
        year_diversity = df.groupby("crop_token")["year"].nunique().to_dict()

        self.crop_stats = {}
        for token, row in raw_stats.iterrows():
            count    = int(row["count"])
            median   = float(row["median"])
            mean     = float(row["mean"])
            std      = float(row["std"]) if not pd.isna(row["std"]) else 0
            n_years  = year_diversity.get(token, 1)

            reg_norm = REGIONAL_NORMS.get(token, median)

            # Смешиваем если мало лет или мало записей
            if n_years == 1:
                # Только один год — нельзя доверять, смешиваем 30/70
                blended = median * 0.3 + reg_norm * 0.7
            elif n_years == 2 or count < 10:
                # Два года или мало записей — 50/50
                blended = median * 0.5 + reg_norm * 0.5
            elif n_years >= 3 and count >= 15:
                # Достаточно данных — доверяем своим данным
                blended = median
            else:
                blended = median * 0.7 + reg_norm * 0.3

            self.crop_stats[token] = {
                "mean":          round(blended, 2),
                "median":        round(median, 2),
                "std":           round(std, 2),
                "count":         count,
                "n_years":       n_years,
                "regional_norm": reg_norm,
            }

        self.dataset_info = {
            "n_records":   len(df),
            "n_fields":    df["field_id"].nunique(),
            "years":       sorted(df["year"].unique().tolist()),
            "n_crops":     df["crop_type"].nunique(),
            "yield_mean":  round(df["yield_t_ha"].mean(), 2),
        }

        # Кодирование
        df = self._encode(df, fit=True)

        # Определяем фичи
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in num_cols
            if c not in self.EXCLUDE_FROM_FEATURES
            and df[c].notna().mean() > 0.15
        ]
        # Добавляем enc колонки явно
        enc_cols = [c for c in df.columns if c.endswith("_enc")
                    and c not in self.EXCLUDE_FROM_FEATURES]
        feature_cols = list(set(feature_cols + enc_cols))
        feature_cols = [c for c in feature_cols if c in df.columns]
        # Убираем weather_year (дубль year)
        feature_cols = [c for c in feature_cols if c not in ("weather_year", "year")]

        # Принудительно включаем метео-фичи если они есть в датасете
        WEATHER_COLS = [
            "spring_precip_mm", "spring_temp_mean", "spring_gdd", "spring_frost_days",
            "veg_precip_mm", "veg_temp_mean", "veg_gdd", "veg_heat_days",
            "aug_precip_mm", "aug_temp_mean",
            "season_precip_mm", "season_gdd_total", "aridity_index",
        ]
        for wc in WEATHER_COLS:
            if wc in df.columns and wc not in feature_cols and df[wc].notna().mean() > 0.3:
                feature_cols.append(wc)

        self.feature_names = sorted(feature_cols)

        mask = df["yield_t_ha"].notna() & (df["yield_t_ha"] >= 0)
        X = df.loc[mask, self.feature_names].fillna(df[self.feature_names].median())
        y = df.loc[mask, "yield_t_ha"]
        years = df.loc[mask, "year"]

        print(f"\n  Фичи ({len(self.feature_names)}): {self.feature_names}")
        print(f"  Обучающая выборка: {len(X)} записей")

        # ── Кросс-валидация по годам (Leave-One-Year-Out) ──
        print("\n  [Кросс-валидация по годам]")
        logo = LeaveOneGroupOut()
        cv_maes, cv_r2s = [], []

        model_cv = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1
        )

        for train_idx, test_idx in logo.split(X, y, groups=years):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if len(y_te) < 3:
                continue
            model_cv.fit(X_tr, y_tr)
            pred = model_cv.predict(X_te)
            cv_maes.append(mean_absolute_error(y_te, pred))
            r2 = r2_score(y_te, pred) if y_te.std() > 0 else float("nan")
            cv_r2s.append(r2)
            test_year = years.iloc[test_idx[0]]
            print(f"    Тест год {test_year}: MAE={mean_absolute_error(y_te, pred):.2f} т/га, n={len(y_te)}")

        self.cv_results = {
            "mae_mean": round(np.nanmean(cv_maes), 3),
            "mae_std":  round(np.nanstd(cv_maes), 3),
            "r2_mean":  round(np.nanmean(cv_r2s), 3),
            "method":   "LeaveOneYearOut",
        }
        print(f"\n  CV итого — MAE: {self.cv_results['mae_mean']:.2f} ± {self.cv_results['mae_std']:.2f} т/га")
        print(f"  CV R²: {self.cv_results['r2_mean']:.3f}")

        # ── Финальное обучение на всех данных ──
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.trained_at = datetime.now().isoformat()

        y_pred = self.model.predict(X)
        print(f"\n  Train MAE: {mean_absolute_error(y, y_pred):.3f} т/га")
        print(f"  Train R²:  {r2_score(y, y_pred):.3f}")

        return self

    def predict(self, input_data: dict) -> dict:
        """Прогноз урожайности для нового поля/сезона."""
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        df = pd.DataFrame([input_data])
        df = self._encode(df, fit=False)

        # Добавляем метео-прогноз для целевого года
        year = input_data.get("year", datetime.now().year)
        try:
            weather = get_forecast_weather_features(target_year=year)
            for k, v in weather.items():
                if k != "weather_source":
                    input_data[k] = v
        except Exception:
            pass

        X = pd.DataFrame(0.0, index=[0], columns=self.feature_names)
        for col in self.feature_names:
            if col in df.columns:
                X[col] = df[col].values[0]

        X = X.fillna(0)
        pred = float(self.model.predict(X)[0])
        pred = max(0.0, pred)

        # Доверительный интервал (по деревьям)
        tree_preds = np.array([t.predict(X)[0] for t in self.model.estimators_])
        ci_low  = float(np.percentile(tree_preds, 10))
        ci_high = float(np.percentile(tree_preds, 90))

        crop_token = input_data.get("crop_token", "")
        crop_avg = self.crop_stats.get(crop_token, {}).get("mean", pred)
        area = input_data.get("seeding_area_ha") or input_data.get("harvest_area_ha") or 1

        # Корректировка по сорту
        variety_name = input_data.get("variety", "")
        variety_factor = 1.0
        variety_note = ""
        if variety_name and variety_name not in ("---", "— не указан —", ""):
            try:
                from external_db import variety_yield_factor
                variety_factor = variety_yield_factor(crop_token, variety_name)
                if variety_factor != 1.0:
                    diff_pct = round((variety_factor - 1) * 100)
                    variety_note = f"{'+' if diff_pct > 0 else ''}{diff_pct}% к базовому прогнозу"
            except ImportError:
                pass

        # Ограничение по агрономическому максимуму культуры
        # Агрономический максимум для Акмолинской / СКО
        # Источник: БНС РК, НПЦ ЗХ им. Бараева, производственная статистика
        CROP_MAX_YIELD = {
            "WHEAT_DURUM":        3.5,   # рекорд хозяйств ~3 т/га
            "WHEAT_WHITE":        3.5,
            "WHEAT_SFT_RD_WTR":   3.5,
            "WHEAT_HRD_RD_WTR":   3.5,
            "WHEAT_EURO_WTR":     4.0,
            "BARLEY":             4.5,
            "BARLEY_EURO_SPR":    4.5,
            "OATS":               4.5,
            "OATS_EURO":          4.5,
            "FLAX":               1.8,
            "CANOLA":             3.0,
            "RAPE_SEED":          3.0,
            "RAPE_SEED_E_OIL":    3.0,
            "PEAS_FIELD":         3.0,
            "PEA_TRAPPER":        3.0,
            "SUNFLOWER_OIL":      3.5,
            "SUNFLOWER_E_OIL":    4.0,
            "CORN_WET":          10.0,
            "CORN_SILAGE":       50.0,
            "GRASS_FORAGE":      30.0,
            "MILLET":             2.0,
            "POTATOES_FOR_RETAIL": 45.0,
        }
        max_yield = CROP_MAX_YIELD.get(crop_token, 6.0)
        # Без достаточного кол-ва лет модель экстраполирует нереалистично
        # Смешиваем прогноз модели с историческим средним по культуре
        n_years_crop = self.crop_stats.get(crop_token, {}).get("n_years", 5)
        if n_years_crop <= 2:
            # Мало лет — доверяем 40% модели, 60% истории
            pred_blended = pred * 0.4 + crop_avg * 0.6 if crop_avg > 0 else pred
        elif n_years_crop == 3:
            # 3 года — 60% модели, 40% истории
            pred_blended = pred * 0.6 + crop_avg * 0.4 if crop_avg > 0 else pred
        else:
            # 4+ лет — доверяем модели, лёгкое смягчение к истории
            pred_blended = pred * 0.75 + crop_avg * 0.25 if crop_avg > 0 else pred

        pred_adj = round(min(max_yield, max(0.0, pred_blended * variety_factor)), 2)
        ci_low_adj   = round(max(0, min(pred_adj, ci_low * variety_factor)), 2)
        ci_high_adj  = round(min(max_yield, max(pred_adj, ci_high * variety_factor * 0.6)), 2)

        return {
            "field_id":         input_data.get("field_id", ""),
            "crop_type":        input_data.get("crop_type", ""),
            "predicted_yield":  pred_adj,
            "predicted_base":   round(pred, 2),      # без коррекции сорта
            "variety_factor":   round(variety_factor, 3),
            "variety_note":     variety_note,
            "ci_low":           ci_low_adj,
            "ci_high":          ci_high_adj,
            "crop_hist_avg":    round(crop_avg, 2),
            "total_yield_est":  round(pred_adj * area, 1),
            "unit":             "т/га",
            "model_mae":        self.cv_results.get("mae_mean"),
        }

    def feature_importance(self, top_n=15):
        imp = self.model.feature_importances_
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"\n  Модель сохранена: {path}")

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self):
        return {
            "trained_at":    self.trained_at,
            "dataset_info":  self.dataset_info,
            "cv_results":    self.cv_results,
            "n_features":    len(self.feature_names),
            "features":      self.feature_names,
            "known_crops":   list(self.crop_stats.keys()),
        }


# ─── FastAPI интеграция ───────────────────────────────────

FASTAPI_CODE = '''
# Добавить в backend/routers/ml.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle, os

router = APIRouter(prefix="/ml", tags=["ML"])
_model = None

def get_model():
    global _model
    if _model is None:
        path = os.environ.get("MODEL_PATH", "models/yield_forecaster_v2.pkl")
        if not os.path.exists(path):
            raise HTTPException(503, detail="Модель не обучена. POST /ml/train")
        with open(path, "rb") as f:
            _model = pickle.load(f)
    return _model


class ForecastRequest(BaseModel):
    field_id: str
    crop_token: str
    crop_type: str
    year: int
    seeding_area_ha: float
    seeding_date: Optional[str] = None
    harvest_date: Optional[str] = None      # ожидаемая дата уборки
    tillage_type: Optional[str] = "Рыхлитель"
    tillage_operations: Optional[int] = 1
    avg_depth_cm: Optional[float] = 25.0
    fertilizer_n_kg_ha: Optional[float] = 0.0
    fertilizer_p_kg_ha: Optional[float] = 0.0
    fertilizer_k_kg_ha: Optional[float] = 0.0
    soil_ph: Optional[float] = 6.5
    soil_humus_pct: Optional[float] = 3.0


@router.post("/forecast")
def forecast(req: ForecastRequest):
    return get_model().predict(req.dict())

@router.get("/model/info")
def model_info():
    return get_model().summary()

@router.get("/model/feature-importance")
def feature_importance():
    return get_model().feature_importance().to_dict("records")

@router.post("/train")
def train():
    """Переобучить модель на всех доступных данных."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "train_all_years.py"],
        capture_output=True, text=True
    )
    global _model
    _model = None  # сбросить кэш
    return {"stdout": result.stdout, "returncode": result.returncode}
'''


# ─── Точка входа ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Обучение модели урожайности")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Папка с JSON файлами")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Папка для сохранения модели")
    args = parser.parse_args()

    print(f"Загрузка данных из: {args.data_dir}")
    df = build_dataset(data_dir=args.data_dir)

    print(f"\nИтоговый датасет: {len(df)} записей, {df['field_id'].nunique()} полей")
    print(f"Годы: {sorted(df['year'].unique())}")
    print(f"Культур: {df['crop_type'].nunique()}")

    model = YieldForecasterV2()
    model.fit(df)

    print("\n[Топ-10 важных фич]")
    print(model.feature_importance(10).to_string(index=False))

    model_path = str(Path(args.model_dir) / "yield_forecaster_v2.pkl")
    model.save(model_path)

    # Тестовый прогноз
    print("\n[Тест: Пшеница твердая, 150 га, 2026]")
    result = model.predict({
        "field_id":          "test-001",
        "crop_token":        "WHEAT_DURUM",
        "crop_type":         "Пшеница (твердая)",
        "year":              2026,
        "seeding_area_ha":   150,
        "seeding_date":      "2026-05-10",
        "harvest_date":      "2026-08-20",
        "tillage_type":      "Рыхлитель",
        "tillage_operations": 1,
        "avg_depth_cm":      25,
        "fertilizer_n_kg_ha": 60,
        "soil_ph":           6.5,
    })
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n✓ Готово!")
