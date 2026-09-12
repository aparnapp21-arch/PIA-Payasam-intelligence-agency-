import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PIA - Payasam Intelligence Agency",
    page_icon="🍮",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/payasam_model.pkl")


# ============================================================
# CUSTOM CSS - LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN PAGE
       ======================================================== */

    .stApp {
        background: linear-gradient(135deg, #fffaf2, #ffffff);
        color: #222222;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #222222;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #222222;
        margin-top: 25px;
        margin-bottom: 10px;
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    .stApp p {
        color: #333333;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    label {
        color: #222222 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #222222 !important;
        font-weight: 500;
    }


    /* ========================================================
       NUMBER INPUT BOX
       ======================================================== */

    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #cccccc !important;
    }

    div[data-baseweb="input"] input {
        color: #222222 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #222222 !important;
    }

    /* Number input + and - buttons */

    div[data-testid="stNumberInput"] button {
        color: #222222 !important;
        background-color: #ffffff !important;
        border: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #f2f2f2 !important;
    }


    /* ========================================================
       SELECT BOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #222222 !important;
    }

    div[data-baseweb="select"] input {
        color: #222222 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    div.stButton > button {
        width: 100%;
        height: 3.2em;
        border-radius: 12px;

        background-color: #ffffff !important;
        color: #222222 !important;

        border: 1px solid #bbbbbb !important;

        font-size: 18px;
        font-weight: 700;

        transition: 0.2s;
    }

    div.stButton > button p {
        color: #222222 !important;
    }

    div.stButton > button:hover {
        background-color: #f5f5f5 !important;
        color: #111111 !important;
        border-color: #999999 !important;
    }


    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {
        padding: 30px;
        border-radius: 20px;

        text-align: center;

        background: #ffffff;

        border: 1px solid #dddddd;

        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.08);

        margin-top: 20px;
    }

    .score-label {
        font-size: 16px;
        color: #666666;
        font-weight: 500;
    }

    .score {
        font-size: 60px;
        font-weight: 800;
        color: #222222;
        margin: 5px 0;
    }

    .score-unit {
        font-size: 25px;
        color: #555555;
    }

    .level {
        font-size: 28px;
        font-weight: 700;
        color: #222222;
        margin-top: 10px;
    }

    .message {
        font-size: 17px;
        font-style: italic;
        color: #555555;
        margin-top: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 13px;
        color: #777777;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border: none;
        border-top: 1px solid #dddddd;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🍮 PIA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Payasam Intelligence Agency</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        font-size:20px;
        color:#444444;
        margin-bottom:30px;
    ">
        Because apparently, even payasam needs AI.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🥄 Payasam Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the recipe parameters and let PIA estimate "
    "the consistency of your payasam."
)


# ============================================================
# ROW 1
# ============================================================

col1, col2 = st.columns(2)

with col1:

    payasam_type = st.selectbox(
        "🍮 Payasam Type",
        [
            "Palada",
            "Semiya",
            "Rice",
            "Parippu",
            "Paal"
        ]
    )

with col2:

    milk_ml = st.number_input(
        "🥛 Milk Quantity (ml)",
        min_value=200,
        max_value=800,
        value=500,
        step=10
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2 = st.columns(2)

with col1:

    water_ml = st.number_input(
        "💧 Water Quantity (ml)",
        min_value=100,
        max_value=600,
        value=250,
        step=10
    )

with col2:

    main_ingredient_g = st.number_input(
        "🌾 Main Ingredient (g)",
        min_value=30,
        max_value=150,
        value=100,
        step=5
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2 = st.columns(2)

with col1:

    sugar_g = st.number_input(
        "🍬 Sugar Quantity (g)",
        min_value=50,
        max_value=200,
        value=120,
        step=5
    )

with col2:

    cooking_time_min = st.number_input(
        "⏱️ Cooking Time (minutes)",
        min_value=15,
        max_value=60,
        value=45,
        step=1
    )


# ============================================================
# ROW 4
# ============================================================

col1, col2 = st.columns(2)

with col1:

    temperature_c = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=60,
        max_value=90,
        value=80,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔮 ANALYZE PAYASAM"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "payasam_type": [payasam_type],
            "milk_ml": [milk_ml],
            "water_ml": [water_ml],
            "main_ingredient_g": [main_ingredient_g],
            "sugar_g": [sugar_g],
            "cooking_time_min": [cooking_time_min],
            "temperature_c": [temperature_c]
        }
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]


    # Convert to normal Python float
    prediction = float(prediction)


    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))


    # --------------------------------------------------------
    # DETERMINE PAYASAM LEVEL
    # --------------------------------------------------------

    if prediction < 40:

        level = "THIN"
        emoji = "🥛"
        message = "Basically payasam-flavoured milk."

    elif prediction < 70:

        level = "MEDIUM"
        emoji = "🥄"
        message = "Decent consistency. PIA approves."

    elif prediction < 85:

        level = "THICK"
        emoji = "🍮"
        message = "Spoon resistance detected."

    else:

        level = "VERY THICK"
        emoji = "🧱"
        message = "Proceed with caution. Spoon may surrender."


    # ========================================================
    # PIA VERDICT
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 PIA Verdict</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    result_html = f"""
    <div class="result-card">

        <div class="score-label">
            PREDICTED CONSISTENCY
        </div>

        <div class="score">
            {prediction:.2f}
            <span class="score-unit">/ 100</span>
        </div>

        <div class="level">
            {emoji} {level}
        </div>

        <div class="message">
            "{message}"
        </div>

    </div>
    """


    st.html(result_html)


    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    st.write("")

    st.progress(
        prediction / 100
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        PIA — Payasam Intelligence Agency<br>

        An unnecessarily sophisticated solution
        to an extremely important problem. 🍮

    </div>
    """,
    unsafe_allow_html=True
)