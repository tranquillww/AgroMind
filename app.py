"""
AgroMind — Прогноз урожайности
Запуск: py -m streamlit run app.py
"""

import sys, pickle, warnings, requests
from datetime import date, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from train_all_years import YieldForecasterV2
from soil_zones import soil_zone_widget
from weather_features import load_nasa_history
from external_db import get_all_regional_stats, variety_input_widget

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AgroMind — Прогноз урожайности", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geologica:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'Geologica',sans-serif;}
.stApp{background:#0c160c;color:#ddeedd;}
[data-testid="stSidebar"]{background:#0a140a!important;border-right:1px solid #182818;}
.main-header{background:linear-gradient(135deg,#172e17,#0d200d);border:1px solid #2a522a;border-radius:20px;padding:28px 36px;margin-bottom:24px;}
.main-header h1{font-size:1.8rem;font-weight:700;color:#6dbc6d;margin:0 0 4px;letter-spacing:-.5px;}
.main-header p{color:#4a7a4a;margin:0;font-size:.88rem;font-weight:300;}
.forecast-card{background:linear-gradient(135deg,#172e17,#122212);border:1px solid #2a522a;border-radius:20px;padding:28px 32px 24px;}
.forecast-label{font-size:.65rem;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#3a6a3a;margin-bottom:14px;}
.forecast-yield{font-size:4rem;font-weight:700;color:#6dbc6d;line-height:1;letter-spacing:-3px;}
.forecast-unit{font-size:1rem;color:#4a7a4a;font-weight:300;margin-top:2px;}
.forecast-sub{font-size:.8rem;color:#3a6a3a;margin-top:10px;line-height:1.6;}
.conf-bar-wrap{background:#0c200c;border-radius:100px;height:6px;margin:12px 0 4px;overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,#e8a020,#6dbc6d);}
.conf-label{font-size:.65rem;color:#2a5a2a;letter-spacing:1px;}
.sec{font-size:.65rem;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:#2a5a2a;padding-bottom:6px;border-bottom:1px solid #182818;margin:20px 0 10px;}
.info-box{background:#0a2218;border:1px solid #183828;border-left:3px solid #3a8a4a;border-radius:10px;padding:14px 16px;font-size:.82rem;color:#4a9a5a;line-height:1.7;}
.weather-card{background:#0a200a;border:1px solid #182818;border-radius:14px;padding:18px 20px;margin-bottom:10px;}
.weather-temp{font-size:2.2rem;font-weight:700;color:#6dbc6d;letter-spacing:-1px;}
.weather-sub{font-size:.78rem;color:#3a6a3a;margin-top:4px;}
.day-card{background:#0c200c;border:1px solid #182818;border-radius:10px;padding:8px 4px;text-align:center;font-size:.72rem;}
.day-dow{color:#2a5a2a;margin-bottom:2px;}
.day-icon{font-size:1.2rem;margin:2px 0;}
.day-temp{color:#6dbc6d;font-weight:600;}
.day-prec{color:#2a6a6a;margin-top:2px;}
.stSelectbox label,.stSlider label,.stNumberInput label,.stDateInput label,.stTextInput label,.stRadio label{color:#4a7a4a!important;font-size:.78rem!important;}
.stButton>button{background:linear-gradient(135deg,#286028,#1a4a1a)!important;color:#b8e8b8!important;border:1px solid #347034!important;border-radius:10px!important;padding:12px 24px!important;font-family:'Geologica',sans-serif!important;font-weight:500!important;font-size:.92rem!important;width:100%!important;transition:all .2s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#347034,#244a24)!important;transform:translateY(-1px)!important;}
div[data-testid="stMetricValue"]{color:#6dbc6d!important;font-size:1.6rem!important;}
div[data-testid="stMetricLabel"]{color:#3a6a3a!important;font-size:.75rem!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CROPS = {
    "Пшеница (твердая)":"WHEAT_DURUM","Пшеница (белозерная)":"WHEAT_WHITE",
    "Пшеница (мягкая краснозерная, озимая)":"WHEAT_SFT_RD_WTR",
    "Пшеница (стекловидная краснозерная, озимая)":"WHEAT_HRD_RD_WTR",
    "Пшеница (Европа, озимая)":"WHEAT_EURO_WTR","Ячмень":"BARLEY",
    "Ячмень (Европа, яровой)":"BARLEY_EURO_SPR","Овес":"OATS","Овес (Европа)":"OATS_EURO",
    "Рапс масличный":"CANOLA","Рапсовое семя":"RAPE_SEED","Подсолнечник":"SUNFLOWER_OIL",
    "Подсолнечник (Европа, масло)":"SUNFLOWER_E_OIL","Полевой горох":"PEAS_FIELD",
    "Горох (посевной)":"PEA_TRAPPER","Лен":"FLAX","Кукуруза":"CORN_WET",
    "Кукурузный силос":"CORN_SILAGE","Просо":"MILLET","Зеленый корм":"GRASS_FORAGE",
}
TILLAGE_TYPES = ["Рыхлитель","Вспашка","Дискование","Чизелевание","Мин. обработка","Без обработки"]
WMO_ICONS = {0:"☀️",1:"🌤️",2:"⛅",3:"☁️",45:"🌫️",51:"🌦️",55:"🌧️",61:"🌧️",65:"🌧️",71:"❄️",75:"❄️",80:"🌦️",82:"⛈️",95:"⛈️"}
DAY_NAMES = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]

@st.cache_resource
def load_model(path="models/yield_forecaster_v2.pkl"):
    if not Path(path).exists(): return None
    with open(path,"rb") as f: return pickle.load(f)

@st.cache_data(ttl=3600)
def fetch_weather(lat=53.28, lon=69.39):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":lat,"longitude":lon,
            "current":"temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily":"temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "forecast_days":7,"timezone":"auto"
        }, timeout=8)
        return r.json() if r.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=86400)
def season_stats(path="nasa_power_history_5y.csv"):
    try:
        df = load_nasa_history(path)
        rows = []
        for yr in sorted(df["year"].unique()):
            if yr > datetime.now().year: continue
            y = df[df["year"]==yr]; veg = y[y["month"].isin([6,7])]
            rows.append({
                "Год":int(yr),
                "Весна мм":int(y[y["month"].isin([4,5])]["precipitation"].sum()),
                "Вегетац мм":int(veg["precipitation"].sum()),
                "Т июн-июл":round(veg["temperature_2m"].mean(),1),
                "Август мм":int(y[y["month"]==8]["precipitation"].sum()),
            })
        return pd.DataFrame(rows).tail(5)
    except:
        return pd.DataFrame()

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def render_weather(lat, lon):
    w = fetch_weather(lat, lon)
    if not w: st.caption("Погода недоступна"); return
    cur = w.get("current", {})
    icon = WMO_ICONS.get(cur.get("weather_code", 0), "🌡️")
    temp = cur.get("temperature_2m", "—")
    hum  = cur.get("relative_humidity_2m", "—")
    wind = cur.get("wind_speed_10m", "—")
    prec = cur.get("precipitation", 0)
    st.markdown(
        '<div class="weather-card">'
        f'<div style="font-size:2rem;margin-bottom:6px;">{icon}</div>'
        f'<div class="weather-temp">{temp}°C</div>'
        f'<div class="weather-sub">💧 {hum}% &nbsp;·&nbsp; 💨 {wind} км/ч &nbsp;·&nbsp; 🌧 {prec} мм</div>'
        '</div>',
        unsafe_allow_html=True
    )
    daily  = w.get("daily", {})
    dates  = daily.get("time", [])[:7]
    t_max  = daily.get("temperature_2m_max", [])
    t_min  = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    codes  = daily.get("weather_code", [])
    if dates:
        cols = st.columns(7)
        for i, d in enumerate(dates):
            dow = DAY_NAMES[datetime.strptime(d, "%Y-%m-%d").weekday()]
            ic  = WMO_ICONS.get(codes[i] if i < len(codes) else 0, "🌡️")
            mx  = f"{t_max[i]:.0f}" if i < len(t_max) else "-"
            mn  = f"{t_min[i]:.0f}" if i < len(t_min) else ""
            pr  = f"{precip[i]:.0f}" if i < len(precip) else "0"
            with cols[i]:
                st.markdown(
                    '<div class="day-card">'
                    f'<div class="day-dow">{dow}</div>'
                    f'<div class="day-icon">{ic}</div>'
                    f'<div class="day-temp">{mx}°</div>'
                    f'<div style="color:#3a6a3a;font-size:.68rem;">{mn}°</div>'
                    f'<div class="day-prec">{pr}мм</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

model = load_model()
st.markdown(
    '<div class="main-header">'
    '<h1>🌾 Прогноз урожайности</h1>'
    '<p>AgroMind &nbsp;·&nbsp; Северный Казахстан &nbsp;·&nbsp; ИИ-прогноз урожайности</p>'
    '</div>',
    unsafe_allow_html=True
)
if model is None:
    st.error("Модель не найдена. Запусти: py train_all_years.py --data-dir data/jd")
    st.stop()

with st.sidebar:
    sec("Поле")
    field_id = st.text_input("ID поля (опционально)", placeholder="напр. field-001")
    year     = st.selectbox("Сезон прогноза", [2026, 2027, 2028])

    sec("Культура")
    crop_name  = st.selectbox("Культура", list(CROPS.keys()))
    crop_token = CROPS[crop_name]
    variety_result = variety_input_widget(st, crop_token, key_prefix="variety")

    sec("Посев")
    area = st.number_input("Площадь (га)", min_value=1.0, max_value=10000.0, value=150.0, step=10.0)
    c1, c2 = st.columns(2)
    with c1: seeding_date = st.date_input("Дата посева",  value=date(year, 5, 10))
    with c2: harvest_date = st.date_input("Ожид. уборка", value=date(year, 8, 20))

    sec("Обработка почвы")
    tillage_type = st.selectbox("Тип обработки", TILLAGE_TYPES)
    c3, c4 = st.columns(2)
    with c3: tillage_ops = st.number_input("Обработок", min_value=0, max_value=5, value=1)
    with c4: depth       = st.number_input("Глубина см", min_value=0, max_value=50, value=25)

    sec("Удобрения (кг/га)")
    c5, c6, c7 = st.columns(3)
    with c5: n_fert = st.number_input("N", min_value=0, max_value=300, value=60)
    with c6: p_fert = st.number_input("P", min_value=0, max_value=300, value=30)
    with c7: k_fert = st.number_input("K", min_value=0, max_value=300, value=20)

    sec("Координаты поля (для погоды)")
    c8, c9 = st.columns(2)
    with c8: field_lat = st.number_input("Широта",  value=53.28, format="%.4f")
    with c9: field_lon = st.number_input("Долгота", value=69.39, format="%.4f")
    st.session_state["_lat"] = field_lat
    st.session_state["_lon"] = field_lon

    soil_result = soil_zone_widget(st, key_prefix="soil")

    st.markdown("---")
    predict_btn = st.button("🌱 Рассчитать прогноз", use_container_width=True)

col_main, col_right = st.columns([3, 2], gap="large")

with col_main:
    if predict_btn:
        st.session_state["ran_predict"] = True
        st.session_state["pred_lat"]    = field_lat
        st.session_state["pred_lon"]    = field_lon
        with st.spinner("Считаю прогноз..."):
            result = model.predict({
                "field_id":           field_id or "—",
                "crop_token":         crop_token,
                "crop_type":          crop_name,
                "year":               year,
                "seeding_area_ha":    area,
                "seeding_date":       str(seeding_date),
                "harvest_date":       str(harvest_date),
                "tillage_type":       tillage_type,
                "tillage_operations": tillage_ops,
                "avg_depth_cm":       depth,
                "fertilizer_n_kg_ha": n_fert,
                "fertilizer_p_kg_ha": p_fert,
                "fertilizer_k_kg_ha": k_fert,
                "soil_ph":            soil_result.get("soil_ph", 7.0),
                "soil_humus_pct":     soil_result.get("soil_humus_pct", 3.5),
                "variety":            variety_result.get("variety", ""),
            })
        st.session_state["last_result"] = result
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "Культура":      crop_name,
            "Год":           year,
            "Площадь га":    area,
            "Прогноз т/га":  result["predicted_yield"],
            "Диапазон":      f"{result['ci_low']:.1f}–{result['ci_high']:.1f}",
            "Валовый сбор т":result["total_yield_est"],
        })

    if "last_result" in st.session_state:
        r     = st.session_state["last_result"]
        pred  = r["predicted_yield"]
        ci_l  = r["ci_low"]
        ci_h  = r["ci_high"]
        hist  = r["crop_hist_avg"]
        total = r["total_yield_est"]
        mae   = r.get("model_mae", 4.3)
        vnote = r.get("variety_note", "")
        diff_pct = ((pred - hist) / hist * 100) if hist > 0 else 0
        diff_str = f"+{diff_pct:.0f}%" if diff_pct >= 0 else f"{diff_pct:.0f}%"
        diff_col = "#6dbc6d" if diff_pct >= 0 else "#f87171"
        sub = f"Диапазон: {ci_l:.1f} — {ci_h:.1f} т/га  ·  Ист. средний: {hist:.2f} т/га  ·  {diff_str} к среднему"
        if vnote:
            sub += f"  ·  Сорт: {vnote}"
        st.markdown(
            '<div class="forecast-card">'
            f'<div class="forecast-label">Прогноз · {crop_name} · {year}</div>'
            f'<div class="forecast-yield">{pred:.2f}</div>'
            '<div class="forecast-unit">т/га</div>'
            f'<div class="forecast-sub" style="color:{diff_col};">{sub}</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Валовый сбор", f"{total:,.0f} т")
        m2.metric("Площадь",     f"{area:,.0f} га")
        m3.metric("Ист. средняя",f"{hist:.2f} т/га")
        m4.metric("Точность",    f"±{mae:.1f} т/га")
        ci_w = max(5, min(95, int((1 - (ci_h - ci_l) / max(ci_h, 0.01)) * 100)))
        st.markdown(
            '<div style="margin-top:16px;">'
            '<div class="conf-label">УВЕРЕННОСТЬ МОДЕЛИ</div>'
            '<div class="conf-bar-wrap">'
            f'<div class="conf-bar-fill" style="width:{ci_w}%;"></div>'
            '</div>'
            '<div class="conf-label">Узкий интервал = выше уверенность</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="border:1px dashed #182818;border-radius:20px;padding:80px 40px;text-align:center;margin-top:8px;">'
            '<div style="font-size:3rem;margin-bottom:12px;">🌱</div>'
            '<div style="font-size:1rem;font-weight:500;color:#2a5a2a;">Заполни параметры в боковой панели</div>'
            '<div style="font-size:.85rem;margin-top:6px;color:#1e3a1e;">и нажми «Рассчитать прогноз»</div>'
            '</div>',
            unsafe_allow_html=True
        )

    if "history" in st.session_state and st.session_state.history:
        sec("История прогнозов")
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Скачать CSV",
            data=df_h.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

with col_right:
    summary   = model.summary()
    info      = summary.get("dataset_info", {})
    cv        = summary.get("cv_results", {})
    years_str = ", ".join(str(y) for y in info.get("years", []))

    if st.session_state.get("ran_predict"):
        sec("Погода на поле")
        render_weather(
            st.session_state.get("pred_lat", 53.28),
            st.session_state.get("pred_lon", 69.39),
        )
        sec("Метео по сезонам (NASA)")
        df_s = season_stats()
        if not df_s.empty:
            st.dataframe(df_s, hide_index=True, use_container_width=True, height=215)
        sec("О модели")
        st.markdown(
            '<div class="info-box">'
            f'Random Forest · NASA POWER + данные полей<br>'
            f'Данные: исторические наблюдения<br>'
            f'Полей: 100+ · Записей: 200+<br>'
            f'Культур: {info.get("n_crops","—")} · MAE: {cv.get("mae_mean","—")} т/га'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        sec("О модели")
        st.markdown(
            '<div class="info-box">'
            '<b>Random Forest</b> · NASA POWER + Open-Meteo<br><br>'
            f'Данные: исторические наблюдения<br>'
            f'Полей: 100+ · Записей: 200+<br>'
            f'Культур: {info.get("n_crops","—")}<br>'
            f'CV MAE: {cv.get("mae_mean","—")} т/га'
            '</div>',
            unsafe_allow_html=True
        )
        sec("Региональная статистика (Акмолинская)")
        st.dataframe(get_all_regional_stats("Акмолинская"), hide_index=True, use_container_width=True, height=290)

