# ============================================================
# INTELLIGENT STUDENT PERFORMANCE PREDICTION SYSTEM
# Streamlit Web Application
# ============================================================

import os
import sys
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# ============================================================
# RECOMMENDATION IMPORT
# ============================================================

try:
    from recommendation import get_recommendations as generate_recommendations
except Exception:
    generate_recommendations = None


# ============================================================
# MODEL PATHS
# ============================================================

REGRESSION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_regression_model.pkl"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_classification_model.pkl"
)

IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

regression_model = None
classification_model = None

if os.path.exists(REGRESSION_MODEL_PATH):
    regression_model = joblib.load(REGRESSION_MODEL_PATH)

if os.path.exists(CLASSIFICATION_MODEL_PATH):
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Intelligent Student Performance Prediction System")

st.write(
    "AI/ML based system for predicting student academic performance "
    "and providing personalized learning recommendations."
)


# ============================================================
# SIDEBAR - MODEL STATUS
# ============================================================

st.sidebar.header("⚙️ Model Status")

if regression_model is not None:
    st.sidebar.success("✅ Regression Model Loaded")
else:
    st.sidebar.error("❌ Regression Model Not Found")

if classification_model is not None:
    st.sidebar.success("✅ Classification Model Loaded")
else:
    st.sidebar.warning("⚠️ Classification Model Not Found")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Student Information")

st.write(
    "Enter the student's academic, family, social and attendance information."
)


with st.form("student_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with col1:

        st.subheader("👤 Personal Information")

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        sex = st.selectbox(
            "Gender",
            ["F", "M"]
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=25,
            value=17
        )

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

        famsize = st.selectbox(
            "Family Size",
            ["LE3", "GT3"]
        )

        Pstatus = st.selectbox(
            "Parent Cohabitation Status",
            ["T", "A"]
        )

        Medu = st.slider(
            "Mother Education",
            0,
            4,
            2
        )

        Fedu = st.slider(
            "Father Education",
            0,
            4,
            2
        )

        Mjob = st.selectbox(
            "Mother Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = st.selectbox(
            "Father Job",
            ["teacher", "health", "services", "at_home", "other"]
        )


    # --------------------------------------------------------
    # ACADEMIC INFORMATION
    # --------------------------------------------------------

    with col2:

        st.subheader("📚 Academic Information")

        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        traveltime = st.slider(
            "Travel Time",
            1,
            4,
            2
        )

        studytime = st.slider(
            "Weekly Study Time",
            1,
            4,
            2
        )

        failures = st.number_input(
            "Past Class Failures",
            min_value=0,
            max_value=4,
            value=0
        )

        schoolsup = st.selectbox(
            "Extra School Support",
            ["yes", "no"]
        )

        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"]
        )

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        activities = st.selectbox(
            "Extra-Curricular Activities",
            ["yes", "no"]
        )

        nursery = st.selectbox(
            "Attended Nursery School",
            ["yes", "no"]
        )


    # --------------------------------------------------------
    # LIFESTYLE INFORMATION
    # --------------------------------------------------------

    with col3:

        st.subheader("🏠 Lifestyle Information")

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"]
        )

        famrel = st.slider(
            "Family Relationship Quality",
            1,
            5,
            4
        )

        freetime = st.slider(
            "Free Time",
            1,
            5,
            3
        )

        goout = st.slider(
            "Going Out Frequency",
            1,
            5,
            3
        )

        Dalc = st.slider(
            "Workday Alcohol Consumption",
            1,
            5,
            1
        )

        Walc = st.slider(
            "Weekend Alcohol Consumption",
            1,
            5,
            1
        )

        health = st.slider(
            "Health Status",
            1,
            5,
            3
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=5
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    submitted = st.form_submit_button(
        "🔮 Predict Student Performance"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    if regression_model is None:

        st.error(
            "Regression model is not available. "
            "Please train the model first."
        )

    else:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = {

            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences
        }


        input_df = pd.DataFrame([input_data])


        # ----------------------------------------------------
        # MATCH MODEL FEATURES
        # ----------------------------------------------------

        if hasattr(
            regression_model,
            "feature_names_in_"
        ):

            input_df = input_df.reindex(
                columns=regression_model.feature_names_in_,
                fill_value=0
            )


        # ----------------------------------------------------
        # PREDICT FINAL SCORE
        # ----------------------------------------------------

        prediction = regression_model.predict(
            input_df
        )[0]


        # Keep score between 0 and 20

        prediction = max(
            0,
            min(20, prediction)
        )


        # ----------------------------------------------------
        # SUPPORT LEVEL
        # ----------------------------------------------------

        if prediction <= 9:

            support_level = "High Support"

        elif prediction <= 13:

            support_level = "Medium Support"

        else:

            support_level = "Low Support"


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.header("📊 Prediction Results")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Predicted Final Score",
                f"{prediction:.2f} / 20"
            )


        with result_col2:

            st.metric(
                "Support Level",
                support_level
            )


        # ----------------------------------------------------
        # SUPPORT MESSAGE
        # ----------------------------------------------------

        if support_level == "High Support":

            st.warning(
                "⚠️ This student may require additional academic support."
            )

        elif support_level == "Medium Support":

            st.info(
                "ℹ️ This student may benefit from additional monitoring and guidance."
            )

        else:

            st.success(
                "✅ This student is currently showing a relatively strong performance level."
            )


        # ====================================================
        # MODEL EXPLANATION
        # ====================================================

        st.divider()

        st.subheader("🔍 Model Explanation")

        if os.path.exists(IMPORTANCE_PATH):

            importance_df = pd.read_csv(
                IMPORTANCE_PATH
            )

            st.write(
                "The following features have the strongest "
                "influence on the model's prediction."
            )

            st.dataframe(
                importance_df.head(10),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Feature importance file is not available yet."
            )


        # ====================================================
        # PERSONALIZED RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.subheader(
            "📚 Personalized Recommendations"
        )


        if generate_recommendations is not None:

            recommendations = generate_recommendations(
               prediction,
               support_level
            )

            for i, recommendation in enumerate(
                recommendations,
                start=1
            ):

                st.write(
                    f"**{i}.** {recommendation}"
                )

        else:

            st.info(
                "Recommendation module is not available."
            )


        # ====================================================
        # STUDENT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Student Prediction Summary"
        )


        summary_data = {

            "Predicted Final Score": round(
                prediction,
                2
            ),

            "Support Level": support_level,

            "Study Time": studytime,

            "Past Failures": failures,

            "Absences": absences,

            "Health": health,

            "Family Relationship": famrel

        }


        summary_df = pd.DataFrame(
            [summary_data]
        )


        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DOWNLOAD SUMMARY
        # ====================================================

        st.subheader(
            "⬇️ Export Prediction Summary"
        )


        csv_data = summary_df.to_csv(
            index=False
        )


        st.download_button(
            label="📥 Download Prediction Summary",
            data=csv_data,
            file_name="student_prediction_summary.csv",
            mime="text/csv"
        )