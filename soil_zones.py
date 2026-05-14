"""
AgroMind — Почвенные зоны Казахстана
Источник: агрохимические исследования почв РК, КазНИИПА, областные агрохимлаборатории
"""

# ══════════════════════════════════════════════════════════
# ПОЧВЕННЫЕ ЗОНЫ АКМОЛИНСКОЙ И СКО ОБЛАСТЕЙ
# ══════════════════════════════════════════════════════════

SOIL_ZONES = {
    # ── Акмолинская область ──────────────────────────────
    "Кокшетау и окрестности": {
        "soil_ph":       7.0,
        "soil_humus_pct": 4.2,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный",
        "description":   "Наиболее плодородные почвы региона. Мощный гумусовый горизонт 40-60 см.",
        "oblast":        "Акмолинская",
        "clay_pct":      30.0,
        "sand_pct":      28.0,
    },
    "Атбасарский район": {
        "soil_ph":       7.1,
        "soil_humus_pct": 3.8,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём южный",
        "description":   "Переходная зона между чернозёмом и каштановыми почвами.",
        "oblast":        "Акмолинская",
        "clay_pct":      28.0,
        "sand_pct":      30.0,
    },
    "Астана (пригород)": {
        "soil_ph":       7.3,
        "soil_humus_pct": 3.2,
        "soil_texture":  "loamy",
        "soil_type":     "Тёмно-каштановая",
        "description":   "Типичная тёмно-каштановая почва. Хороша для зерновых при достаточном увлажнении.",
        "oblast":        "Акмолинская",
        "clay_pct":      26.0,
        "sand_pct":      34.0,
    },
    "Степногорский район": {
        "soil_ph":       7.5,
        "soil_humus_pct": 2.8,
        "soil_texture":  "loamy",
        "soil_type":     "Тёмно-каштановая карбонатная",
        "description":   "Карбонатные почвы, требуют внимания к pH при выборе культур.",
        "oblast":        "Акмолинская",
        "clay_pct":      24.0,
        "sand_pct":      36.0,
    },
    "Ерейментауский район": {
        "soil_ph":       7.7,
        "soil_humus_pct": 2.3,
        "soil_texture":  "loamy",
        "soil_type":     "Каштановая",
        "description":   "Засушливая зона. Ограниченный потенциал без орошения.",
        "oblast":        "Акмолинская",
        "clay_pct":      22.0,
        "sand_pct":      40.0,
    },
    "Буландынский район": {
        "soil_ph":       6.9,
        "soil_humus_pct": 4.5,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный мощный",
        "description":   "Одни из лучших почв области. Высокий потенциал урожайности зерновых.",
        "oblast":        "Акмолинская",
        "clay_pct":      31.0,
        "sand_pct":      26.0,
    },
    "Аршалынский район": {
        "soil_ph":       7.2,
        "soil_humus_pct": 3.5,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём южный",
        "description":   "Хорошие почвы для пшеницы и ячменя.",
        "oblast":        "Акмолинская",
        "clay_pct":      27.0,
        "sand_pct":      32.0,
    },
    "Жаркаинский район": {
        "soil_ph":       7.6,
        "soil_humus_pct": 2.5,
        "soil_texture":  "loamy",
        "soil_type":     "Тёмно-каштановая",
        "description":   "Южная зона области, засушливее. Подходит для подсолнечника.",
        "oblast":        "Акмолинская",
        "clay_pct":      23.0,
        "sand_pct":      38.0,
    },
    "Целиноградский район": {
        "soil_ph":       7.1,
        "soil_humus_pct": 3.9,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём южный",
        "description":   "Продуктивные почвы пригородного района.",
        "oblast":        "Акмолинская",
        "clay_pct":      28.0,
        "sand_pct":      30.0,
    },
    "Шортандинский район": {
        "soil_ph":       6.8,
        "soil_humus_pct": 4.8,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный",
        "description":   "Район НПЦ им. Бараева. Эталонные черноземы Северного Казахстана.",
        "oblast":        "Акмолинская",
        "clay_pct":      32.0,
        "sand_pct":      25.0,
    },
    # ── СКО ─────────────────────────────────────────────
    "Петропавловск и окрестности": {
        "soil_ph":       6.7,
        "soil_humus_pct": 5.2,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём выщелоченный",
        "description":   "Лучшие почвы СКО. Высокое содержание гумуса, хорошая структура.",
        "oblast":        "СКО",
        "clay_pct":      33.0,
        "sand_pct":      24.0,
    },
    "Мамлютский район (СКО)": {
        "soil_ph":       6.9,
        "soil_humus_pct": 4.8,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный",
        "description":   "Плодородные черноземы. Высокая продуктивность зерновых.",
        "oblast":        "СКО",
        "clay_pct":      31.0,
        "sand_pct":      26.0,
    },
    "Кызылжарский район (СКО)": {
        "soil_ph":       7.0,
        "soil_humus_pct": 4.5,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный",
        "description":   "Стабильно высокие урожаи пшеницы и ячменя.",
        "oblast":        "СКО",
        "clay_pct":      30.0,
        "sand_pct":      28.0,
    },
    "Тайыншинский район (СКО)": {
        "soil_ph":       7.2,
        "soil_humus_pct": 3.8,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём южный",
        "description":   "Хорошие почвы для зерновых и масличных культур.",
        "oblast":        "СКО",
        "clay_pct":      27.0,
        "sand_pct":      31.0,
    },
    # ── Другие регионы ───────────────────────────────────
    "Костанайская область": {
        "soil_ph":       6.9,
        "soil_humus_pct": 4.3,
        "soil_texture":  "loamy",
        "soil_type":     "Чернозём обыкновенный",
        "description":   "Зерновой пояс Казахстана. Высокопродуктивные черноземы.",
        "oblast":        "Костанайская",
        "clay_pct":      30.0,
        "sand_pct":      27.0,
    },
    "Павлодарская область": {
        "soil_ph":       7.8,
        "soil_humus_pct": 2.2,
        "soil_texture":  "loamy",
        "soil_type":     "Каштановая",
        "description":   "Засушливая зона. Требует орошения для высоких урожаев.",
        "oblast":        "Павлодарская",
        "clay_pct":      21.0,
        "sand_pct":      42.0,
    },
    "Другой район": {
        "soil_ph":       7.0,
        "soil_humus_pct": 3.5,
        "soil_texture":  "loamy",
        "soil_type":     "Средние показатели",
        "description":   "Используйте ручной ввод для уточнения параметров.",
        "oblast":        "—",
        "clay_pct":      27.0,
        "sand_pct":      32.0,
    },
}

# Группировка по области для UI
OBLAST_ZONES = {
    "Акмолинская": [k for k,v in SOIL_ZONES.items() if v["oblast"] == "Акмолинская"],
    "СКО":         [k for k,v in SOIL_ZONES.items() if v["oblast"] == "СКО"],
    "Другие":      [k for k,v in SOIL_ZONES.items() if v["oblast"] not in ("Акмолинская", "СКО")],
}


def get_soil_by_zone(zone_name: str) -> dict:
    """Возвращает почвенные параметры для зоны."""
    zone = SOIL_ZONES.get(zone_name, SOIL_ZONES["Другой район"])
    return {
        "soil_ph":        zone["soil_ph"],
        "soil_humus_pct": zone["soil_humus_pct"],
        "soil_texture":   zone["soil_texture"],
        "clay_pct":       zone["clay_pct"],
        "sand_pct":       zone["sand_pct"],
        "soil_type":      zone["soil_type"],
        "description":    zone["description"],
        "source":         "kz_soil_zones",
    }


def soil_zone_widget(st, key_prefix: str = "zone") -> dict:
    """
    Streamlit виджет выбора почвенной зоны.
    Возвращает dict с параметрами почвы.
    """
    # Способ ввода
    mode = st.radio(
        "Способ ввода почвы",
        ["🗺️ По зоне РК", "✍️ Вручную"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )

    if mode == "🗺️ По зоне РК":
        # Выбор области
        oblast = st.selectbox(
            "Область",
            list(OBLAST_ZONES.keys()),
            key=f"{key_prefix}_oblast",
        )
        # Выбор района
        zone_name = st.selectbox(
            "Район / зона",
            OBLAST_ZONES[oblast],
            key=f"{key_prefix}_zone",
        )
        zone_data = SOIL_ZONES[zone_name]
        soil = get_soil_by_zone(zone_name)

        # Карточка зоны
        st.markdown(
            '<div style="background:#0a2218;border:1px solid #183828;border-left:3px solid #3a8a4a;'
            'border-radius:8px;padding:12px 14px;font-size:0.8rem;color:#4a9a5a;line-height:1.7;margin-top:4px;">'
            f'<b style="color:#6dbc6d">{zone_data["soil_type"]}</b><br>'
            f'pH: {soil["soil_ph"]} &nbsp;·&nbsp; Гумус: {soil["soil_humus_pct"]}%<br>'
            f'{zone_data["description"]}'
            '</div>',
            unsafe_allow_html=True,
        )

        # Возможность скорректировать
        with st.expander("Скорректировать вручную"):
            soil["soil_ph"] = st.slider(
                "pH", 4.0, 9.0,
                value=float(soil["soil_ph"]), step=0.1,
                key=f"{key_prefix}_ph_adj",
            )
            soil["soil_humus_pct"] = st.slider(
                "Гумус (%)", 0.5, 10.0,
                value=float(soil["soil_humus_pct"]), step=0.1,
                key=f"{key_prefix}_humus_adj",
            )
        return soil

    else:
        # Ручной ввод
        ph    = st.slider("pH почвы", 4.0, 9.0, value=7.0, step=0.1, key=f"{key_prefix}_ph")
        humus = st.slider("Гумус (%)", 0.5, 10.0, value=3.5, step=0.1, key=f"{key_prefix}_humus")
        texture = st.selectbox(
            "Текстура",
            ["loamy", "clay_loam", "clay", "sandy"],
            format_func=lambda x: {"loamy":"Суглинок","clay_loam":"Глинисто-суглинистая","clay":"Глинистая","sandy":"Песчаная"}[x],
            key=f"{key_prefix}_texture",
        )
        return {
            "soil_ph":        ph,
            "soil_humus_pct": humus,
            "soil_texture":   texture,
            "clay_pct":       None,
            "sand_pct":       None,
            "source":         "manual",
        }


if __name__ == "__main__":
    print("Почвенные зоны Казахстана:")
    for zone, data in SOIL_ZONES.items():
        print(f"  {zone}: pH={data['soil_ph']}, гумус={data['soil_humus_pct']}%, {data['soil_type']}")
