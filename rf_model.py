"""
AgroMind — Random Forest модель прогноза урожайности
Обучение, кросс-валидация, сохранение, предсказание
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. ТРЕНИРОВКА МОДЕЛИ
# ─────────────────────────────────────────────

class YieldForecaster:
    """
    Random Forest модель прогноза урожайности.
    Поддерживает обучение по нескольким годам и предсказание на новый сезон.
    """

    def __init__(self, model_type: str = "rf"):
        """
        model_type: "rf" — Random Forest, "gb" — Gradient Boosting
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.label_encoders = {}
        self.crop_stats = {}       # средняя урожайность по культуре (fallback)
        self.trained_at = None
        self.cv_scores = {}

    def _build_model(self):
        if self.model_type == "gb":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Кодирует категориальные колонки."""
        df = df.copy()
        cat_cols = ["crop_token", "crop_type", "tillage_type", "soil_texture", "variety"]

        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[f"{col}_enc"] = df[col].fillna("unknown").astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает финальный набор фич."""
        df = df.copy()

        # Временные фичи из дат
        for date_col, prefix in [("seeding_date", "seed"), ("harvest_date", "harv"), ("tillage_date", "till")]:
            if date_col in df.columns:
                col = pd.to_datetime(df[date_col], utc=True, errors="coerce")
                df[f"{prefix}_doy"]   = col.dt.day_of_year
                df[f"{prefix}_month"] = col.dt.month

        # Длина вегетации
        if "seeding_date" in df.columns and "harvest_date" in df.columns:
            sd = pd.to_datetime(df["seeding_date"], utc=True, errors="coerce")
            hd = pd.to_datetime(df["harvest_date"], utc=True, errors="coerce")
            df["growing_days"] = (hd - sd).dt.days

        return df

    def fit(self, df: pd.DataFrame, target_col: str = "yield_t_ha"):
        """
        Обучает модель на историческом датасете.
        
        df — результат data_pipeline.build_feature_dataset()
        """
        print(f"[YieldForecaster] Обучение на {len(df)} записях...")

        # Сохраняем статистику по культурам (для fallback предсказаний)
        self.crop_stats = df.groupby("crop_token")[target_col].agg(
            ["mean", "std", "count"]
        ).to_dict("index")

        # Кодирование
        df = self._encode_categoricals(df, fit=True)
        df = self._extract_features(df)

        # Определяем фичи автоматически
        # Исключаем: таргет, ID поля, сырые даты, категории до кодирования,
        # и POST-HARVEST данные (утечка: при прогнозе они недоступны)
        exclude = {
            target_col, "field_id", "org_id",
            "seeding_date", "harvest_date", "tillage_date", "forage_date",
            "crop_type", "crop_token", "tillage_type", "soil_texture", "variety",
            "last_timestamp",
            # Данные, доступные только ПОСЛЕ уборки — утечка!
            "total_yield_t", "moisture_pct", "dry_matter_pct",
            "harvest_area_ha", "fuel_l_ha",
        }
        feature_cols = [c for c in df.columns
                        if c not in exclude
                        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                        and df[c].notna().sum() > len(df) * 0.3]

        self.feature_names = feature_cols

        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col]

        # Убираем строки без таргета
        mask = y.notna() & (y > 0)
        X, y = X[mask], y[mask]

        print(f"[YieldForecaster] Фичи ({len(feature_cols)}): {feature_cols}")
        print(f"[YieldForecaster] Обучающая выборка: {len(X)} записей")

        # ─── Кросс-валидация ───
        self.model = self._build_model()
        kf = KFold(n_splits=min(5, len(X) // 5), shuffle=True, random_state=42)

        cv_mae  = cross_val_score(self.model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        cv_r2   = cross_val_score(self.model, X, y, cv=kf, scoring="r2")

        self.cv_scores = {
            "mae_mean":  round(-cv_mae.mean(), 3),
            "mae_std":   round(cv_mae.std(), 3),
            "r2_mean":   round(cv_r2.mean(), 3),
            "r2_std":    round(cv_r2.std(), 3),
            "n_folds":   kf.n_splits,
            "n_samples": len(X),
        }

        print(f"\n[CV результаты]")
        print(f"  MAE:  {self.cv_scores['mae_mean']:.3f} ± {self.cv_scores['mae_std']:.3f} т/га")
        print(f"  R²:   {self.cv_scores['r2_mean']:.3f} ± {self.cv_scores['r2_std']:.3f}")

        # ─── Финальная тренировка на всех данных ───
        self.model.fit(X, y)
        self.trained_at = datetime.now().isoformat()

        # Train-set метрики (для справки)
        y_pred = self.model.predict(X)
        train_mae = mean_absolute_error(y, y_pred)
        train_r2  = r2_score(y, y_pred)
        print(f"\n[Train метрики]")
        print(f"  MAE:  {train_mae:.3f} т/га")
        print(f"  R²:   {train_r2:.3f}")

        return self

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Возвращает важность фич."""
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        imp = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)

        return df_imp

    def predict(self, input_data: dict) -> dict:
        """
        Предсказывает урожайность для нового сезона.
        
        input_data — dict с параметрами поля:
        {
            "field_id": "...",
            "crop_token": "WHEAT_HRD_RD_WTR",
            "crop_type": "Пшеница (твердая)",
            "seeding_date": "2025-05-01",
            "seeding_area_ha": 150,
            "tillage_type": "Рыхлитель",
            "tillage_operations": 1,
            "avg_depth_cm": 25,
            # метео и удобрения — если есть
            "fertilizer_n_kg_ha": 60,
            "soil_ph": 6.5,
            ...
        }
        """
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        df = pd.DataFrame([input_data])
        df = self._encode_categoricals(df, fit=False)
        df = self._extract_features(df)

        # Заполняем отсутствующие фичи медианой (из обучающих данных)
        X = pd.DataFrame(columns=self.feature_names)
        X = pd.concat([X, df[
            [c for c in self.feature_names if c in df.columns]
        ]], ignore_index=True)

        # Добавляем отсутствующие колонки как NaN → 0
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names].fillna(0)

        pred = self.model.predict(X)[0]

        # Получаем доверительный интервал через деревья
        tree_preds = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        ci_low  = np.percentile(tree_preds, 10)
        ci_high = np.percentile(tree_preds, 90)

        # Fallback через статистику культуры
        crop_token = input_data.get("crop_token", "")
        crop_mean = self.crop_stats.get(crop_token, {}).get("mean", pred)

        return {
            "field_id":         input_data.get("field_id", ""),
            "crop_type":        input_data.get("crop_type", ""),
            "predicted_yield":  round(pred, 2),
            "ci_low":           round(max(0, ci_low), 2),
            "ci_high":          round(ci_high, 2),
            "crop_avg_yield":   round(crop_mean, 2),
            "total_yield_est":  round(pred * input_data.get("seeding_area_ha", 1), 1),
            "unit":             "т/га",
        }

    def save(self, path: str):
        """Сохраняет модель в pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[YieldForecaster] Модель сохранена: {path}")

    @staticmethod
    def load(path: str) -> "YieldForecaster":
        """Загружает модель из pickle."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[YieldForecaster] Модель загружена: {path}")
        return model

    def get_summary(self) -> dict:
        """Возвращает сводку о модели."""
        return {
            "model_type":    self.model_type,
            "trained_at":    self.trained_at,
            "n_features":    len(self.feature_names),
            "features":      self.feature_names,
            "cv_scores":     self.cv_scores,
            "known_crops":   list(self.crop_stats.keys()),
        }


# ─────────────────────────────────────────────
# 2. FastAPI ENDPOINT (для интеграции)
# ─────────────────────────────────────────────

FASTAPI_ROUTER_CODE = '''
# Добавить в ваш FastAPI backend

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.insert(0, ".")
from rf_model import YieldForecaster

router = APIRouter(prefix="/ml", tags=["ML Forecast"])

# Загружаем модель при старте
_model: Optional[YieldForecaster] = None

def get_model() -> YieldForecaster:
    global _model
    if _model is None:
        try:
            _model = YieldForecaster.load("models/yield_forecaster.pkl")
        except FileNotFoundError:
            raise HTTPException(503, "Модель не обучена. Запустите /ml/train сначала.")
    return _model


class ForecastRequest(BaseModel):
    field_id: str
    crop_token: str
    crop_type: str
    seeding_area_ha: float
    seeding_date: Optional[str] = None
    tillage_type: Optional[str] = "Рыхлитель"
    tillage_operations: Optional[int] = 1
    avg_depth_cm: Optional[float] = 25.0
    fertilizer_n_kg_ha: Optional[float] = 0.0
    fertilizer_p_kg_ha: Optional[float] = 0.0
    fertilizer_k_kg_ha: Optional[float] = 0.0
    soil_ph: Optional[float] = 6.5
    soil_humus_pct: Optional[float] = 3.0


class ForecastResponse(BaseModel):
    field_id: str
    crop_type: str
    predicted_yield: float
    ci_low: float
    ci_high: float
    crop_avg_yield: float
    total_yield_est: float
    unit: str


@router.post("/forecast", response_model=ForecastResponse)
def forecast_yield(req: ForecastRequest):
    """Прогноз урожайности для поля на следующий сезон."""
    model = get_model()
    result = model.predict(req.dict())
    return ForecastResponse(**result)


@router.post("/train")
def train_model():
    """Переобучить модель на актуальных данных."""
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_feature_dataset
    
    # Укажи пути к своим JSON файлам
    df = build_feature_dataset(
        harvest_files={
            2021: "data/jd/harvest_2021.json",
            2022: "data/jd/harvest_2022.json",
            2023: "data/jd/harvest_2023.json",
            2024: "data/jd/harvest_2024.json",
        },
        seeding_files={
            2022: "data/jd/seeding_2022.json",
            2023: "data/jd/seeding_2023.json",
            2024: "data/jd/seeding_2024.json",
        },
        tillage_files={
            2021: "data/jd/tillage_2021.json",
            2022: "data/jd/tillage_2022.json",
            2023: "data/jd/tillage_2023.json",
            2024: "data/jd/tillage_2024.json",
        },
    )
    
    global _model
    _model = YieldForecaster(model_type="rf")
    _model.fit(df)
    _model.save("models/yield_forecaster.pkl")
    
    return {"status": "ok", "summary": _model.get_summary()}


@router.get("/model/info")
def model_info():
    """Информация о текущей модели."""
    return get_model().get_summary()


@router.get("/model/feature-importance")
def feature_importance():
    """Важность фич."""
    return get_model().feature_importance().to_dict("records")
'''


# ─────────────────────────────────────────────
# 3. ПРИМЕР ЗАПУСКА
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_feature_dataset

    print("=" * 55)
    print("  AgroMind — Yield Forecaster")
    print("=" * 55)

    # 1. Загружаем данные
    df = build_feature_dataset(
        harvest_files={2024: "data/jd/harvest_2024.json"},
        seeding_files={2024: "data/jd/seeding_2024.json"},
        tillage_files={2024: "data/jd/tillage_2024.json"},
    )

    # 2. Обучаем модель
    model = YieldForecaster(model_type="rf")
    model.fit(df)

    # 3. Сохраняем
    model.save("models/yield_forecaster.pkl")

    # 4. Важность фич
    print("\n[Важность фич]")
    print(model.feature_importance().to_string(index=False))

    # 5. Тестовый прогноз
    print("\n[Тестовый прогноз]")
    result = model.predict({
        "field_id":          "example-field-001",
        "crop_token":        "WHEAT_HRD_RD_WTR",
        "crop_type":         "Пшеница (твердая)",
        "seeding_area_ha":   150,
        "seeding_date":      "2025-05-10",
        "harvest_date":      "2025-08-20",
        "tillage_type":      "Рыхлитель",
        "tillage_operations": 1,
        "avg_depth_cm":      25,
        "fertilizer_n_kg_ha": 60,
        "fertilizer_p_kg_ha": 30,
        "soil_ph":           6.5,
        "soil_humus_pct":    3.2,
    })
    for k, v in result.items():
        print(f"  {k}: {v}")
