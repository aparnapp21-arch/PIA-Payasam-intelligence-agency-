import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
import tempfile


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="PIA – Payasam Intelligence Agency",
    page_icon="🥣",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("models/payasam_model.pkl")


# =========================================================
# RECIPE-BASED PREDICTION
# =========================================================

def predict_consistency(
    payasam_type,
    milk_ml,
    water_ml,
    main_ingredient_g,
    sugar_g,
    cooking_time_min,
    temperature_c
):

    input_data = pd.DataFrame([{
        "payasam_type": payasam_type,
        "milk_ml": milk_ml,
        "water_ml": water_ml,
        "main_ingredient_g": main_ingredient_g,
        "sugar_g": sugar_g,
        "cooking_time_min": cooking_time_min,
        "temperature_c": temperature_c
    }])

    prediction = model.predict(input_data)[0]

    return prediction


# =========================================================
# VIDEO MOTION CALCULATION
# =========================================================

def calculate_motion(frame1, frame2):

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        gray1,
        gray2
    )

    motion_score = difference.mean()

    return motion_score


# =========================================================
# REFERENCE IMAGE COMPARISON
# =========================================================

def compare_with_reference(
    frame,
    reference_frame
):

    frame = cv2.resize(
        frame,
        (300, 300)
    )

    reference_frame = cv2.resize(
        reference_frame,
        (300, 300)
    )

    hsv_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hsv_reference = cv2.cvtColor(
        reference_frame,
        cv2.COLOR_BGR2HSV
    )

    hist_frame = cv2.calcHist(
        [hsv_frame],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256]
    )

    hist_reference = cv2.calcHist(
        [hsv_reference],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist_frame,
        hist_frame
    )

    cv2.normalize(
        hist_reference,
        hist_reference
    )

    similarity = cv2.compareHist(
        hist_frame,
        hist_reference,
        cv2.HISTCMP_CORREL
    )

    return similarity


# =========================================================
# VERIFY PAYASAM VIDEO
# =========================================================

def verify_payasam_video(video_path):

    reference_path = "data/middle_frame.jpg"

    reference_frame = cv2.imread(
        reference_path
    )

    if reference_frame is None:
        return False, 0

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return False, 0

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        cap.release()
        return False, 0

    frame_positions = np.linspace(
        0,
        frame_count - 1,
        5
    ).astype(int)

    similarities = []

    for position in frame_positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position)
        )

        success, frame = cap.read()

        if success:

            similarity = compare_with_reference(
                frame,
                reference_frame
            )

            similarities.append(
                similarity
            )

    cap.release()

    if len(similarities) == 0:
        return False, 0

    average_similarity = np.mean(
        similarities
    )

    threshold = 0.30

    is_payasam = (
        average_similarity >= threshold
    )

    return (
        is_payasam,
        average_similarity
    )


# =========================================================
# VIDEO CONSISTENCY ANALYSIS
# =========================================================

def analyze_payasam_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return None

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

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

        motion_scores.append(
            motion
        )

        previous_frame = current_frame

    cap.release()

    if len(motion_scores) == 0:
        return None

    average_motion = np.mean(
        motion_scores
    )

    minimum_motion = np.min(
        motion_scores
    )

    maximum_motion = np.max(
        motion_scores
    )

    motion_variation = np.std(
        motion_scores
    )

    video_score = (
        average_motion * 10
        +
        motion_variation * 5
    )

    video_score = max(
        0,
        min(100, video_score)
    )

    if video_score < 40:

        video_verdict = "THIN 🥛"

    elif video_score < 70:

        video_verdict = "MEDIUM 🥄"

    elif video_score < 85:

        video_verdict = "THICK 🍮"

    else:

        video_verdict = "VERY THICK 🧱"

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "average_motion": average_motion,
        "minimum_motion": minimum_motion,
        "maximum_motion": maximum_motion,
        "motion_variation": motion_variation,
        "video_score": video_score,
        "video_verdict": video_verdict
    }


# =========================================================
# PHOTO ANALYSIS FUNCTIONS
# =========================================================

def get_center_crop(image):

    height, width = image.shape[:2]

    crop_size = int(
        min(height, width) * 0.65
    )

    center_x = width // 2
    center_y = height // 2

    x1 = max(
        0,
        center_x - crop_size // 2
    )

    y1 = max(
        0,
        center_y - crop_size // 2
    )

    x2 = min(
        width,
        center_x + crop_size // 2
    )

    y2 = min(
        height,
        center_y + crop_size // 2
    )

    return image[y1:y2, x1:x2]


def analyze_photo_features(photo):

    # -----------------------------------------------------
    # Resize image
    # -----------------------------------------------------

    photo = cv2.resize(
        photo,
        (400, 400)
    )

    # -----------------------------------------------------
    # Focus mainly on the central food region
    # -----------------------------------------------------

    food_region = get_center_crop(
        photo
    )

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        food_region,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # Smooth image slightly
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # Edge detection
    # -----------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    edge_density = (
        np.mean(edges > 0) * 100
    )

    # -----------------------------------------------------
    # Texture measurement
    # -----------------------------------------------------

    texture_variance = np.var(
        gray
    )

    # Normalize texture approximately to 0–100
    texture_level = min(
        100,
        texture_variance / 20
    )

    # -----------------------------------------------------
    # Brightness
    # -----------------------------------------------------

    brightness = np.mean(
        gray
    )

    # -----------------------------------------------------
    # Saturation
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        food_region,
        cv2.COLOR_BGR2HSV
    )

    saturation = np.mean(
        hsv[:, :, 1]
    )

    # -----------------------------------------------------
    # Calculate smoothness
    # -----------------------------------------------------

    smoothness = (
        100 - edge_density
    )

    smoothness = max(
        0,
        min(100, smoothness)
    )

    # -----------------------------------------------------
    # Payasam likelihood
    #
    # Smooth/liquid food:
    #   high smoothness
    #   low edge density
    #
    # Dry/high-texture food:
    #   lower smoothness
    #   higher edge density
    # -----------------------------------------------------

    smoothness_score = smoothness

    texture_score = (
        100 - texture_level
    )

    texture_score = max(
        0,
        min(100, texture_score)
    )

    # -----------------------------------------------------
    # Reference image similarity
    # -----------------------------------------------------

    reference_path = "data/middle_frame.jpg"

    reference_frame = cv2.imread(
        reference_path
    )

    if reference_frame is not None:

        similarity = compare_with_reference(
            photo,
            reference_frame
        )

        # Histogram correlation can be from -1 to 1
        similarity_normalized = (
            (similarity + 1) / 2
        ) * 100

        similarity_normalized = max(
            0,
            min(100, similarity_normalized)
        )

    else:

        similarity = 0

        similarity_normalized = 0

    # -----------------------------------------------------
    # Combined payasam likelihood
    # -----------------------------------------------------

    payasam_likelihood = (
        smoothness_score * 0.45
        +
        texture_score * 0.35
        +
        similarity_normalized * 0.20
    )

    payasam_likelihood = max(
        0,
        min(100, payasam_likelihood)
    )

    return {
        "edge_density": edge_density,
        "texture_level": texture_level,
        "brightness": brightness,
        "saturation": saturation,
        "smoothness": smoothness,
        "reference_similarity": similarity,
        "payasam_likelihood": payasam_likelihood
    }


def classify_photo(features):

    likelihood = features[
        "payasam_likelihood"
    ]

    edge_density = features[
        "edge_density"
    ]

    texture_level = features[
        "texture_level"
    ]

    # -----------------------------------------------------
    # Reject strongly textured/dry foods
    # -----------------------------------------------------

    if (
        edge_density > 25
        and texture_level > 35
    ):

        return False

    # -----------------------------------------------------
    # Main threshold
    # -----------------------------------------------------

    if likelihood >= 55:

        return True

    return False


# =========================================================
# TITLE
# =========================================================

st.title(
    "🥣 PIA – Payasam Intelligence Agency"
)

st.write(
    "### AI-powered Payasam Consistency Analysis"
)

st.write(
    "Predict payasam consistency using recipe parameters, "
    "photos, or videos."
)

st.divider()


# =========================================================
# RECIPE-BASED ANALYSIS
# =========================================================

st.header(
    "🍚 Recipe-Based Consistency Prediction"
)

col1, col2 = st.columns(2)


with col1:

    payasam_type = st.selectbox(
        "Payasam Type",
        [
            "Rice Payasam",
            "Semiya Payasam",
            "Ada Payasam",
            "Parippu Payasam",
            "Other"
        ]
    )

    milk_ml = st.number_input(
        "Milk (ml)",
        min_value=0,
        max_value=5000,
        value=500
    )

    water_ml = st.number_input(
        "Water (ml)",
        min_value=0,
        max_value=5000,
        value=200
    )

    main_ingredient_g = st.number_input(
        "Main Ingredient (g)",
        min_value=0,
        max_value=2000,
        value=100
    )


with col2:

    sugar_g = st.number_input(
        "Sugar (g)",
        min_value=0,
        max_value=1000,
        value=100
    )

    cooking_time_min = st.number_input(
        "Cooking Time (min)",
        min_value=1,
        max_value=300,
        value=30
    )

    temperature_c = st.number_input(
        "Temperature (°C)",
        min_value=0,
        max_value=200,
        value=90
    )


if st.button(
    "🔮 PREDICT CONSISTENCY",
    use_container_width=True
):

    try:

        prediction = predict_consistency(
            payasam_type,
            milk_ml,
            water_ml,
            main_ingredient_g,
            sugar_g,
            cooking_time_min,
            temperature_c
        )

        prediction = max(
            0,
            min(100, prediction)
        )

        st.subheader(
            "📊 Prediction Result"
        )

        st.metric(
            "Consistency Score",
            f"{prediction:.2f}"
        )

        if prediction < 40:

            st.error(
                "🥛 THIN"
            )

        elif prediction < 70:

            st.warning(
                "🥄 MEDIUM"
            )

        elif prediction < 85:

            st.success(
                "🍮 THICK"
            )

        else:

            st.success(
                "🧱 VERY THICK"
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


st.divider()


# =========================================================
# VIDEO ANALYSIS
# =========================================================

st.header(
    "📹 Video-Based Payasam Analysis"
)

st.write(
    "Upload a video of payasam being stirred or poured."
)

uploaded_video = st.file_uploader(
    "Upload Payasam Video",
    type=[
        "mp4",
        "avi",
        "mov"
    ],
    key="payasam_video"
)


if uploaded_video is not None:

    st.video(
        uploaded_video
    )

    st.info(
        "🔎 The video will first be checked against "
        "the reference payasam image."
    )

    if st.button(
        "🔍 ANALYZE VIDEO",
        use_container_width=True
    ):

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(
                uploaded_video.read()
            )

            temp_video_path = (
                temp_file.name
            )

        # -------------------------------------------------
        # PAYASAM VERIFICATION
        # -------------------------------------------------

        with st.spinner(
            "🔎 Checking whether the video contains payasam..."
        ):

            is_payasam, similarity = (
                verify_payasam_video(
                    temp_video_path
                )
            )

        st.subheader(
            "🔎 Payasam Verification"
        )

        st.metric(
            "Visual Similarity",
            f"{similarity:.2f}"
        )

        # -------------------------------------------------
        # REJECT NON-PAYASAM
        # -------------------------------------------------

        if not is_payasam:

            st.error(
                "❌ This video does not appear to contain payasam."
            )

            st.warning(
                "Please upload a video showing payasam "
                "being stirred or poured."
            )

            st.stop()

        # -------------------------------------------------
        # PAYASAM DETECTED
        # -------------------------------------------------

        st.success(
            "✅ Payasam video detected!"
        )

        # -------------------------------------------------
        # CONSISTENCY ANALYSIS
        # -------------------------------------------------

        with st.spinner(
            "🥣 Analyzing payasam consistency..."
        ):

            result = analyze_payasam_video(
                temp_video_path
            )

        if result is not None:

            st.subheader(
                "📊 Video Consistency Result"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Consistency Score",
                    f"{result['video_score']:.2f}"
                )

            with col2:

                st.metric(
                    "Average Motion",
                    f"{result['average_motion']:.2f}"
                )

            with col3:

                st.metric(
                    "Motion Variation",
                    f"{result['motion_variation']:.2f}"
                )

            st.success(
                f"### Verdict: {result['video_verdict']}"
            )

            st.write(
                f"**Frames:** {result['frame_count']}"
            )

            st.write(
                f"**FPS:** {result['fps']:.2f}"
            )

            st.write(
                f"**Duration:** {result['duration']:.2f} seconds"
            )

            st.write(
                f"**Minimum Motion:** "
                f"{result['minimum_motion']:.2f}"
            )

            st.write(
                f"**Maximum Motion:** "
                f"{result['maximum_motion']:.2f}"
            )

            st.info(
                "ℹ️ Video analysis uses motion/flow characteristics "
                "as a proxy for consistency. It is not a direct "
                "laboratory viscosity measurement."
            )


# =========================================================
# PHOTO ANALYSIS
# =========================================================

st.divider()

st.header(
    "📷 Payasam Photo Analysis"
)

st.write(
    "Upload a clear photo of payasam for visual analysis."
)

uploaded_photo = st.file_uploader(
    "Upload Payasam Photo",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="payasam_photo"
)


if uploaded_photo is not None:

    st.image(
        uploaded_photo,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 ANALYZE PHOTO",
        use_container_width=True,
        key="analyze_photo"
    ):

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                uploaded_photo.read()
            ),
            dtype=np.uint8
        )

        photo = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if photo is None:

            st.error(
                "❌ Unable to read the uploaded image."
            )

        else:

            # -------------------------------------------------
            # ANALYZE IMAGE
            # -------------------------------------------------

            features = analyze_photo_features(
                photo
            )

            is_payasam = classify_photo(
                features
            )

            # -------------------------------------------------
            # DISPLAY FOOD VERIFICATION
            # -------------------------------------------------

            st.subheader(
                "🔎 Payasam Verification"
            )

            st.metric(
                "Payasam Likelihood",
                f"{features['payasam_likelihood']:.2f}%"
            )

            if not is_payasam:

                st.error(
                    "❌ This image does not appear to contain payasam."
                )

                st.warning(
                    "Please upload a clear image of payasam."
                )

                st.write(
                    "The image appears to have visual "
                    "characteristics inconsistent with a "
                    "smooth/liquid payasam."
                )

            else:

                st.success(
                    "✅ Payasam-like image detected!"
                )

                # -------------------------------------------------
                # PHOTO CONSISTENCY ANALYSIS
                # -------------------------------------------------

                st.subheader(
                    "📊 Photo Analysis Result"
                )

                smoothness = features[
                    "smoothness"
                ]

                texture_level = features[
                    "texture_level"
                ]

                brightness = features[
                    "brightness"
                ]

                st.metric(
                    "Visual Smoothness",
                    f"{smoothness:.2f}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Brightness:** "
                        f"{brightness:.2f}"
                    )

                with col2:

                    st.write(
                        f"**Texture Level:** "
                        f"{texture_level:.2f}"
                    )

                with col3:

                    st.write(
                        f"**Edge Density:** "
                        f"{features['edge_density']:.2f}"
                    )

                # -------------------------------------------------
                # CONSISTENCY SCORE
                # -------------------------------------------------

                photo_score = smoothness

                photo_score = max(
                    0,
                    min(100, photo_score)
                )

                st.metric(
                    "Visual Consistency Score",
                    f"{photo_score:.2f}"
                )

                if photo_score < 40:

                    st.error(
                        "🥛 THIN"
                    )

                elif photo_score < 70:

                    st.warning(
                        "🥄 MEDIUM"
                    )

                elif photo_score < 85:

                    st.success(
                        "🍮 THICK"
                    )

                else:

                    st.success(
                        "🧱 VERY THICK"
                    )

                st.info(
                    "ℹ️ Photo analysis estimates consistency "
                    "from visual smoothness, texture and "
                    "reference-image characteristics. "
                    "It is not a direct laboratory viscosity measurement."
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        🥣 <b>PIA – Payasam Intelligence Agency</b><br>
        A deliberately useless AI project for estimating payasam consistency.
    </div>
    """,
    unsafe_allow_html=True
)