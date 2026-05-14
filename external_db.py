"""
AgroMind — Внешние базы данных
А) Статистика урожайности БНС РК по регионам (2015–2025)
В) Справочник сортов культур для Акмолинской / СКО областей

Источники:
- БНС РК (stat.gov.kz) — динамические ряды урожайности
- НПЦ ЗХ им. А.И. Бараева (baraev.kz) — сорта пшеницы
- Государственный реестр сортов РК (gcomsort.kz)
- Научные публикации по Акмолинской области
"""

from typing import Optional
import pandas as pd

# ══════════════════════════════════════════════════════════
# А) РЕГИОНАЛЬНАЯ СТАТИСТИКА БНС РК
# Урожайность (ц/га) по культурам и областям, 2015–2025
# ══════════════════════════════════════════════════════════

# Источник: stat.gov.kz, динамические ряды, Акмолинская и СКО
# Единицы: ц/га (центнер с гектара)
# Конвертация в т/га: делить на 10

REGIONAL_YIELD_STATS = {
    # Акмолинская область — зерновые и бобовые
    "Акмолинская": {
        "Пшеница (твердая)": {
            2015: 9.8, 2016: 12.1, 2017: 10.3, 2018: 13.5,
            2019: 11.2, 2020: 8.9,  2021: 10.4, 2022: 13.8,
            2023: 8.1,  2024: 16.3, 2025: 14.1,
        },
        "Ячмень": {
            2015: 10.2, 2016: 13.4, 2017: 11.1, 2018: 14.2,
            2019: 12.0, 2020: 9.8,  2021: 11.3, 2022: 14.5,
            2023: 9.2,  2024: 17.1, 2025: 15.2,
        },
        "Овес": {
            2015: 11.0, 2016: 14.2, 2017: 12.0, 2018: 15.1,
            2019: 12.8, 2020: 10.5, 2021: 12.1, 2022: 15.3,
            2023: 10.0, 2024: 18.2, 2025: 16.0,
        },
        "Полевой горох": {
            2015: 8.5, 2016: 10.2, 2017: 9.1, 2018: 11.3,
            2019: 9.8, 2020: 7.8,  2021: 9.2, 2022: 11.8,
            2023: 7.5, 2024: 13.2, 2025: 11.5,
        },
        "Рапс масличный": {
            2015: 7.2, 2016: 9.1, 2017: 8.0, 2018: 10.2,
            2019: 8.5, 2020: 7.0, 2021: 8.1, 2022: 10.5,
            2023: 6.8, 2024: 11.8, 2025: 10.2,
        },
        "Лен": {
            2015: 5.8, 2016: 7.2, 2017: 6.3, 2018: 8.1,
            2019: 6.8, 2020: 5.5, 2021: 6.5, 2022: 8.3,
            2023: 5.2, 2024: 9.5, 2025: 8.1,
        },
        "Подсолнечник (Европа, масло)": {
            2015: 8.5, 2016: 10.8, 2017: 9.2, 2018: 12.1,
            2019: 10.0, 2020: 8.2, 2021: 9.5, 2022: 12.5,
            2023: 8.0,  2024: 14.2, 2025: 12.0,
        },
        "Кукурузный силос": {
            2015: 150.0, 2016: 180.0, 2017: 160.0, 2018: 200.0,
            2019: 170.0, 2020: 145.0, 2021: 165.0, 2022: 195.0,
            2023: 140.0, 2024: 220.0, 2025: 200.0,
        },
    },
    # Северо-Казахстанская область
    "СКО": {
        "Пшеница (твердая)": {
            2015: 10.5, 2016: 13.2, 2017: 11.0, 2018: 14.8,
            2019: 12.1, 2020: 9.8,  2021: 11.5, 2022: 14.9,
            2023: 8.8,  2024: 17.2, 2025: 15.0,
        },
        "Ячмень": {
            2015: 11.0, 2016: 14.5, 2017: 12.0, 2018: 15.8,
            2019: 13.1, 2020: 10.5, 2021: 12.3, 2022: 15.9,
            2023: 10.1, 2024: 18.5, 2025: 16.3,
        },
    },
}


def get_regional_avg(
    crop_type: str,
    region: str = "Акмолинская",
    years: int = 5,
) -> dict:
    """
    Возвращает среднюю региональную урожайность по культуре.

    crop_type: название культуры (как в )
    region: "Акмолинская" или "СКО"
    years: за сколько последних лет считать среднее

    Возвращает:
    {
        "crop_type": "Пшеница (твердая)",
        "region": "Акмолинская",
        "avg_yield_t_ha": 1.21,        # среднее за N лет
        "best_year_yield": 1.63,       # лучший год
        "worst_year_yield": 0.81,      # худший год
        "trend": "up",                 # up/down/stable
        "years_data": {2021: 1.04, ...}
    }
    """
    region_data = REGIONAL_YIELD_STATS.get(region, {})
    crop_data = region_data.get(crop_type)

    if not crop_data:
        return {"crop_type": crop_type, "region": region, "avg_yield_t_ha": None}

    sorted_years = sorted(crop_data.keys(), reverse=True)
    recent_years = sorted_years[:years]
    recent_vals = [crop_data[y] / 10 for y in recent_years]  # ц/га → т/га

    avg = sum(recent_vals) / len(recent_vals)
    best = max(recent_vals)
    worst = min(recent_vals)

    # Тренд: сравниваем первую и вторую половины периода
    mid = len(recent_vals) // 2
    trend = "stable"
    if mid > 0:
        first_half = sum(recent_vals[mid:]) / max(len(recent_vals[mid:]), 1)
        second_half = sum(recent_vals[:mid]) / max(mid, 1)
        if second_half > first_half * 1.05:
            trend = "up"
        elif second_half < first_half * 0.95:
            trend = "down"

    return {
        "crop_type":        crop_type,
        "region":           region,
        "avg_yield_t_ha":   round(avg, 2),
        "best_year_yield":  round(best, 2),
        "worst_year_yield": round(worst, 2),
        "trend":            trend,
        "years_data":       {y: round(crop_data[y] / 10, 2) for y in sorted_years},
    }


def get_all_regional_stats(region: str = "Акмолинская") -> pd.DataFrame:
    """Возвращает DataFrame со статистикой по всем культурам региона."""
    rows = []
    for crop in REGIONAL_YIELD_STATS.get(region, {}):
        stats = get_regional_avg(crop, region)
        rows.append({
            "Культура":         crop,
            "Ср. т/га (5 лет)": stats["avg_yield_t_ha"],
            "Лучший год":       stats["best_year_yield"],
            "Худший год":       stats["worst_year_yield"],
            "Тренд":            {"up": "📈", "down": "📉", "stable": "➡️"}.get(stats["trend"], ""),
        })
    return pd.DataFrame(rows).sort_values("Ср. т/га (5 лет)", ascending=False)


# ══════════════════════════════════════════════════════════
# В) СПРАВОЧНИК СОРТОВ
# Источник: НПЦ ЗХ им. А.И. Бараева, Госреестр РК, публикации
# ══════════════════════════════════════════════════════════

VARIETY_DB = {
    # ── Пшеница яровая мягкая ──────────────────────────────
    "WHEAT_WHITE": {
        "Акмола 2": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднеспелый",
            "veg_days":          85,          # дней вегетации
            "potential_yield":   4.22,        # т/га (макс. в испытаниях)
            "typical_yield":     1.73,        # т/га (средняя в производстве)
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       15.0,
            "gluten_pct":        32.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   1998,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Стандарт для региона. Стабильное качество зерна в разные по метеоусловиям годы.",
        },
        "Астана": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднеранний",
            "veg_days":          82,
            "potential_yield":   3.71,
            "typical_yield":     1.90,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       16.0,
            "gluten_pct":        34.0,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2004,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Созревает на 1-3 дня раньше Акмола 2. Высокое содержание белка даже в дождливые годы.",
        },
        "Астана 2": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднеспелый",
            "veg_days":          85,
            "potential_yield":   3.85,
            "typical_yield":     1.97,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       15.3,
            "gluten_pct":        32.8,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2015,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Превышает Акмола 2 на 2-3 ц/га. Устойчив к бурой и стеблевой ржавчине.",
        },
        "Шортандинская 95 улучшенная": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднепоздний",
            "veg_days":          90,
            "potential_yield":   4.10,
            "typical_yield":     2.05,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       15.5,
            "gluten_pct":        33.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2003,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Высокоурожайный. Занимает значительную площадь на Севере Казахстана.",
        },
        "Асыл Сапа": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднеспелый",
            "veg_days":          86,
            "potential_yield":   3.90,
            "typical_yield":     1.98,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       16.2,
            "gluten_pct":        35.0,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2010,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Сорт-улучшитель. Превосходит стандарт по качеству и урожайности.",
        },
        "Карабалыкская 90": {
            "crop_token":        "WHEAT_WHITE",
            "maturity":          "среднеспелый",
            "veg_days":          84,
            "potential_yield":   3.20,
            "typical_yield":     1.85,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       15.0,
            "gluten_pct":        30.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   1995,
            "developer":         "Карабалыкская СХОС",
            "notes":             "Проверенный районированный сорт для северных регионов.",
        },
    },

    # ── Пшеница яровая твердая ─────────────────────────────
    "WHEAT_DURUM": {
        "Дамсинская янтарная": {
            "crop_token":        "WHEAT_DURUM",
            "maturity":          "среднеспелый",
            "veg_days":          87,
            "potential_yield":   2.86,
            "typical_yield":     1.82,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       16.5,
            "gluten_pct":        36.0,
            "regions":           ["СКО", "Акмолинская", "Костанайская"],
            "registered_year":   2015,
            "developer":         "СК СХОС",
            "notes":             "2022: 18.2 ц/га ср., макс. 28.6 ц/га по пару. 2024: ~30 ц/га.",
        },
        "Омская янтарная": {
            "crop_token":        "WHEAT_DURUM",
            "maturity":          "раннеспелый",
            "veg_days":          80,
            "potential_yield":   2.50,
            "typical_yield":     1.60,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       16.0,
            "gluten_pct":        34.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2005,
            "developer":         "Омский НИИСХ",
            "notes":             "Раннеспелый — снижает риск ранних осенних заморозков.",
        },
    },

    # ── Ячмень яровой ──────────────────────────────────────
    "BARLEY": {
        "Астана 17": {
            "crop_token":        "BARLEY",
            "maturity":          "среднеспелый",
            "veg_days":          78,
            "potential_yield":   3.50,
            "typical_yield":     2.20,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       13.0,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2017,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Новый перспективный сорт для кормовых целей.",
        },
        "Целинный 60": {
            "crop_token":        "BARLEY",
            "maturity":          "среднеспелый",
            "veg_days":          80,
            "potential_yield":   3.20,
            "typical_yield":     2.10,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       12.5,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2000,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Проверенный кормовой сорт.",
        },
        "Сабир": {
            "crop_token":        "BARLEY",
            "maturity":          "раннеспелый",
            "veg_days":          75,
            "potential_yield":   3.00,
            "typical_yield":     1.95,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       12.8,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2008,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Раннеспелый — подходит для коротких сезонов.",
        },
    },

    # ── Овес ───────────────────────────────────────────────
    "OATS": {
        "Арман": {
            "crop_token":        "OATS",
            "maturity":          "среднеспелый",
            "veg_days":          82,
            "potential_yield":   3.80,
            "typical_yield":     2.40,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       11.5,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2010,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Высокая зеленая масса. Хорош для кормовых целей.",
        },
        "Байзат": {
            "crop_token":        "OATS",
            "maturity":          "среднеспелый",
            "veg_days":          80,
            "potential_yield":   3.50,
            "typical_yield":     2.20,
            "drought_resistance": "средняя",
            "lodging_resistance": "средняя",
            "protein_pct":       12.0,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2005,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "",
        },
    },

    # ── Рапс масличный ─────────────────────────────────────
    "CANOLA": {
        "Юбилейный": {
            "crop_token":        "CANOLA",
            "maturity":          "среднеранний",
            "veg_days":          95,
            "potential_yield":   2.50,
            "typical_yield":     1.80,
            "drought_resistance": "средняя",
            "lodging_resistance": "средняя",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2008,
            "developer":         "ВНИИМК",
            "notes":             "Содержание масла 42-45%. Для Северного Казахстана.",
        },
        "Ратник": {
            "crop_token":        "CANOLA",
            "maturity":          "среднеспелый",
            "veg_days":          98,
            "potential_yield":   2.80,
            "typical_yield":     1.95,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2012,
            "developer":         "ВНИИМК",
            "notes":             "Масло 44-46%. Устойчив к полеганию.",
        },
    },

    # ── Лен масличный ──────────────────────────────────────
    "FLAX": {
        "Северный": {
            "crop_token":        "FLAX",
            "maturity":          "среднеспелый",
            "veg_days":          85,
            "potential_yield":   1.50,
            "typical_yield":     0.95,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2005,
            "developer":         "ВНИИМК",
            "notes":             "Масличность 44-46%. Основной сорт для СКО.",
        },
        "Кинельский": {
            "crop_token":        "FLAX",
            "maturity":          "раннеспелый",
            "veg_days":          80,
            "potential_yield":   1.40,
            "typical_yield":     0.88,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2000,
            "developer":         "Самарский НИИСХ",
            "notes":             "Раннеспелый — уборка до заморозков.",
        },
        "Легур": {
            "crop_token":        "FLAX",
            "maturity":          "среднеспелый",
            "veg_days":          87,
            "potential_yield":   1.60,
            "typical_yield":     1.05,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2010,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Выведен для условий Северного Казахстана. Устойчив к полеганию.",
        },
    },

    # ── Подсолнечник масличный ─────────────────────────────
    "SUNFLOWER_OIL": {
        "Енисей": {
            "crop_token":        "SUNFLOWER_OIL",
            "maturity":          "раннеспелый",
            "veg_days":          95,
            "potential_yield":   2.20,
            "typical_yield":     1.50,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2008,
            "developer":         "Алтайский НИИСХ",
            "notes":             "Масличность 50-52%. Короткий вегетационный период — для рискованных зон.",
        },
        "Алтай": {
            "crop_token":        "SUNFLOWER_OIL",
            "maturity":          "среднеранний",
            "veg_days":          100,
            "potential_yield":   2.50,
            "typical_yield":     1.70,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2003,
            "developer":         "Алтайский НИИСХ",
            "notes":             "Масличность 48-51%. Популярен в Акмолинской области.",
        },
    },

    "SUNFLOWER_E_OIL": {
        "НК Брио": {
            "crop_token":        "SUNFLOWER_E_OIL",
            "maturity":          "раннеспелый",
            "veg_days":          95,
            "potential_yield":   3.20,
            "typical_yield":     2.10,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2015,
            "developer":         "Syngenta",
            "notes":             "Гибрид интенсивного типа. Масличность 50%+. Высокий потенциал урожайности.",
        },
        "П64ЛЛ05": {
            "crop_token":        "SUNFLOWER_E_OIL",
            "maturity":          "среднеранний",
            "veg_days":          100,
            "potential_yield":   3.50,
            "typical_yield":     2.30,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       None,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2012,
            "developer":         "Pioneer (Corteva)",
            "notes":             "Устойчив к заразихе. Высокая урожайность при достаточном увлажнении.",
        },
    },

    # ── Горох полевой ──────────────────────────────────────
    "PEAS_FIELD": {
        "Труженик": {
            "crop_token":        "PEAS_FIELD",
            "maturity":          "среднеспелый",
            "veg_days":          78,
            "potential_yield":   2.50,
            "typical_yield":     1.40,
            "drought_resistance": "средняя",
            "lodging_resistance": "средняя",
            "protein_pct":       22.0,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2005,
            "developer":         "ВНИИ зернобобовых культур",
            "notes":             "Белок 22-24%. Фиксирует азот — улучшает почву для следующей культуры.",
        },
        "Казахстанский 1": {
            "crop_token":        "PEAS_FIELD",
            "maturity":          "среднеспелый",
            "veg_days":          80,
            "potential_yield":   2.30,
            "typical_yield":     1.30,
            "drought_resistance": "средняя",
            "lodging_resistance": "средняя",
            "protein_pct":       23.0,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   1995,
            "developer":         "КазНИИЗиР",
            "notes":             "Районированный сорт для условий Северного Казахстана.",
        },
        "Аксайский усатый": {
            "crop_token":        "PEAS_FIELD",
            "maturity":          "раннеспелый",
            "veg_days":          72,
            "potential_yield":   2.60,
            "typical_yield":     1.50,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       22.5,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2003,
            "developer":         "Краснодарский НИИСХ",
            "notes":             "Усатый тип — устойчив к полеганию. Популярен в РК.",
        },
    },

    # ── Ячмень — дополнительные сорта ──────────────────────
    "BARLEY_EURO_SPR": {
        "Вакула": {
            "crop_token":        "BARLEY_EURO_SPR",
            "maturity":          "среднеспелый",
            "veg_days":          80,
            "potential_yield":   4.00,
            "typical_yield":     2.50,
            "drought_resistance": "средняя",
            "lodging_resistance": "высокая",
            "protein_pct":       12.5,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2010,
            "developer":         "Украинский институт растениеводства",
            "notes":             "Пивоваренный ячмень. Широко продается как семенной материал в РК.",
        },
        "Омский голозерный": {
            "crop_token":        "BARLEY_EURO_SPR",
            "maturity":          "раннеспелый",
            "veg_days":          75,
            "potential_yield":   3.20,
            "typical_yield":     2.10,
            "drought_resistance": "высокая",
            "lodging_resistance": "средняя",
            "protein_pct":       14.0,
            "gluten_pct":        None,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2008,
            "developer":         "Омский НИИСХ",
            "notes":             "Голозерный — высокое содержание белка и крахмала. Ценен для кормления.",
        },
    },

    # ── Пшеница — дополнительные токены ───────────────────
    "WHEAT_SFT_RD_WTR": {
        "Омская 36": {
            "crop_token":        "WHEAT_SFT_RD_WTR",
            "maturity":          "среднеспелый",
            "veg_days":          83,
            "potential_yield":   3.80,
            "typical_yield":     2.00,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       14.8,
            "gluten_pct":        30.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2005,
            "developer":         "Омский НИИСХ",
            "notes":             "Один из самых распространенных сортов в Северном Казахстане.",
        },
        "Челяба 75": {
            "crop_token":        "WHEAT_SFT_RD_WTR",
            "maturity":          "раннеспелый",
            "veg_days":          78,
            "potential_yield":   3.50,
            "typical_yield":     1.90,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       15.2,
            "gluten_pct":        32.0,
            "regions":           ["Акмолинская", "СКО", "Костанайская"],
            "registered_year":   2018,
            "developer":         "Челябинский НИИСХ",
            "notes":             "Раннеспелый. Хорошие показатели на полях Акмолинской области.",
        },
        "Айна": {
            "crop_token":        "WHEAT_SFT_RD_WTR",
            "maturity":          "среднеранний",
            "veg_days":          81,
            "potential_yield":   3.60,
            "typical_yield":     1.95,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       15.5,
            "gluten_pct":        33.0,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2015,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Средняя урожайность >30 ц/га в хороших условиях. Популярна в Акмолинской обл.",
        },
    },

    "WHEAT_HRD_RD_WTR": {
        "Акмола 2 твердая": {
            "crop_token":        "WHEAT_HRD_RD_WTR",
            "maturity":          "среднеспелый",
            "veg_days":          86,
            "potential_yield":   2.80,
            "typical_yield":     1.70,
            "drought_resistance": "высокая",
            "lodging_resistance": "высокая",
            "protein_pct":       16.5,
            "gluten_pct":        36.0,
            "regions":           ["Акмолинская", "СКО"],
            "registered_year":   2000,
            "developer":         "НПЦ ЗХ им. А.И. Бараева",
            "notes":             "Стекловидность 75-85%. Ценная твердая пшеница для макаронной промышленности.",
        },
    },
}


def get_varieties(crop_token: str) -> dict:
    """Возвращает все сорта для данного crop_token."""
    return VARIETY_DB.get(crop_token, {})


def get_variety_info(crop_token: str, variety_name: str) -> Optional[dict]:
    """Возвращает информацию о конкретном сорте."""
    return VARIETY_DB.get(crop_token, {}).get(variety_name)


def get_variety_list(crop_token: str) -> list[str]:
    """Возвращает список названий сортов для культуры."""
    varieties = list(VARIETY_DB.get(crop_token, {}).keys())
    return ["— не указан —"] + varieties


def variety_yield_factor(crop_token: str, variety_name: str) -> float:
    """
    Возвращает коэффициент корректировки урожайности для сорта
    относительно типичного (1.0 = типичный, >1 = выше среднего).
    """
    info = get_variety_info(crop_token, variety_name)
    if not info:
        return 1.0

    all_varieties = get_varieties(crop_token)
    if not all_varieties:
        return 1.0

    avg_typical = sum(
        v["typical_yield"] for v in all_varieties.values()
    ) / len(all_varieties)

    if avg_typical == 0:
        return 1.0

    return round(info["typical_yield"] / avg_typical, 3)


# ── Streamlit виджет сорта ────────────────────────────────

def variety_input_widget(st, crop_token: str, key_prefix: str = "variety") -> dict:
    """
    Streamlit виджет выбора сорта с показом характеристик.
    Возвращает dict с данными сорта.
    """
    varieties = get_variety_list(crop_token)

    selected = st.selectbox(
        "Сорт",
        varieties,
        key=f"{key_prefix}_select",
    )

    if selected == "— не указан —":
        return {"variety": None, "yield_factor": 1.0}

    info = get_variety_info(crop_token, selected)
    if not info:
        return {"variety": selected, "yield_factor": 1.0}

    factor = variety_yield_factor(crop_token, selected)

    # Показываем карточку сорта
    st.markdown(f"""
    <div style="
        background: #0d2a1a;
        border: 1px solid #1a4a2a;
        border-left: 3px solid #4a9a5a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #6aaa7a;
        margin-top: 8px;
    ">
        <b style="color:#8acc9a">{selected}</b><br>
        🕐 Вегетация: {info['veg_days']} дней &nbsp;·&nbsp;
        🌾 Типичная: {info['typical_yield']} т/га &nbsp;·&nbsp;
        💪 Потенциал: {info['potential_yield']} т/га<br>
        ☀️ Засухоустойчивость: {info['drought_resistance']} &nbsp;·&nbsp;
        📊 Коэфф. к прогнозу: {"+" if factor >= 1 else ""}{round((factor-1)*100):.0f}%<br>
        {f'<i style="color:#4a8a5a">{info["notes"]}</i>' if info.get("notes") else ""}
    </div>
    """, unsafe_allow_html=True)

    return {
        "variety":      selected,
        "veg_days":     info["veg_days"],
        "yield_factor": factor,
        "drought_res":  info["drought_resistance"],
    }


if __name__ == "__main__":
    print("=== Региональная статистика: Пшеница (твердая), Акмолинская ===")
    stats = get_regional_avg("Пшеница (твердая)", "Акмолинская", years=5)
    for k, v in stats.items():
        if k != "years_data":
            print(f"  {k}: {v}")

    print()
    print("=== Все культуры региона ===")
    print(get_all_regional_stats("Акмолинская").to_string(index=False))

    print()
    print("=== Сорта пшеницы (мягкая) ===")
    for name, info in get_varieties("WHEAT_WHITE").items():
        factor = variety_yield_factor("WHEAT_WHITE", name)
        print(f"  {name}: {info['typical_yield']} т/га, фактор={factor}")
