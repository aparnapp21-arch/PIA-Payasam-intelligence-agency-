import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
import tempfile


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
# VIDEO ANALYSIS FUNCTIONS
# ============================================================

def calculate_motion(frame1, frame2):

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray1, gray2)

    motion_score = difference.mean()

    return motion_score


def analyze_payasam_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = frame_count / fps if fps > 0 else 0

    motion_scores = []

    success, previous_frame = cap.read()

    while success:

        success, current_frame = cap.read()

        if not success:
            break

        motion = calculate_motion(
            previous_frame,
            current_frame
        )

        motion_scores.append(motion)

        previous_frame = current_frame

    cap.release()

    if not motion_scores:
        return None

    average_motion = float(np.mean(motion_scores))
    minimum_motion = float(np.min(motion_scores))
    maximum_motion = float(np.max(motion_scores))
    motion_variation = float(np.std(motion_scores))

    # Video-based consistency score
    video_score = (average_motion * 10) + (motion_variation * 5)

    # Keep score between 0 and 100
    video_score = max(0, min(100, video_score))

    # Determine consistency
    if video_score < 40:
        video_verdict = "THIN"
        video_emoji = "🥛"

    elif video_score < 70:
        video_verdict = "MEDIUM"
        video_emoji = "🥄"

    elif video_score < 85:
        video_verdict = "THICK"
        video_emoji = "🍮"

    else:
        video_verdict = "VERY THICK"
        video_emoji = "🧱"

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "average_motion": average_motion,
        "minimum_motion": minimum_motion,
        "maximum_motion": maximum_motion,
        "motion_variation": motion_variation,
        "video_score": video_score,
        "video_verdict": video_verdict,
        "video_emoji": video_emoji
    }


# ============================================================
# CUSTOM CSS - LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #fffaf2, #ffffff);
        color: #222222;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

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

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #222222;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .stApp p {
        color: #333333;
    }

    label {
        color: #222222 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #222222 !important;
        font-weight: 500;
    }

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

    div[data-testid="stNumberInput"] button {
        color: #222222 !important;
        background-color: #ffffff !important;
        border: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #f2f2f2 !important;
    }

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

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 13px;
        color: #777777;
    }

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
# RECIPE-BASED ANALYSIS
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
# RECIPE PREDICTION
# ============================================================

if predict_button:

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

    prediction = model.predict(input_data)[0]

    prediction = float(prediction)

    prediction = max(0, min(100, prediction))


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


    st.markdown(
        '<div class="section-title">🔍 PIA Verdict</div>',
        unsafe_allow_html=True
    )

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

    st.write("")

    st.progress(prediction / 100)


# ============================================================
# DIVIDER
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Video-Based Consistency Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a short video of the payasam while stirring or flowing. "
    "PIA will analyze the motion in the video and estimate consistency."
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "📹 Upload Payasam Video",
    type=["mp4", "avi", "mov"]
)


# ============================================================
# VIDEO ANALYSIS BUTTON
# ============================================================

if uploaded_video is not None:

    st.video(uploaded_video)

    analyze_video_button = st.button(
        "🎥 ANALYZE VIDEO"
    )

    if analyze_video_button:

        with st.spinner("PIA is analyzing the payasam..."):

            # Create temporary video file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getbuffer()
                )

                temp_video_path = temp_file.name


            # Analyze video
            video_result = analyze_payasam_video(
                temp_video_path
            )


        if video_result is None:

            st.error(
                "Unable to analyze the video. Please try another video."
            )

        else:

            # ------------------------------------------------
            # VIDEO RESULT
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">🔬 Video Analysis Result</div>',
                unsafe_allow_html=True
            )


            result_html = f"""
            <div class="result-card">

                <div class="score-label">
                    VIDEO CONSISTENCY SCORE
                </div>

                <div class="score">
                    {video_result["video_score"]:.2f}
                    <span class="score-unit">/ 100</span>
                </div>

                <div class="level">
                    {video_result["video_emoji"]}
                    {video_result["video_verdict"]}
                </div>

                <div class="message">
                    Video-based consistency estimate
                </div>

            </div>
            """

            st.html(result_html)


            # ------------------------------------------------
            # VIDEO PARAMETERS
            # ------------------------------------------------

            st.write("")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Average Motion",
                    f'{video_result["average_motion"]:.2f}'
                )

            with col2:

                st.metric(
                    "Motion Variation",
                    f'{video_result["motion_variation"]:.2f}'
                )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Minimum Motion",
                    f'{video_result["minimum_motion"]:.2f}'
                )

            with col2:

                st.metric(
                    "Maximum Motion",
                    f'{video_result["maximum_motion"]:.2f}'
                )


            st.write("")

            st.info(
                "This is a video-based consistency estimate derived "
                "from visual motion. It is a proxy and not a direct "
                "laboratory measurement of viscosity."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        PIA — Payasam Intelligence Agency

        An unnecessarily sophisticated solution
        to an extremely important problem. 🍮

    </div>
    """,
    unsafe_allow_html=True
)