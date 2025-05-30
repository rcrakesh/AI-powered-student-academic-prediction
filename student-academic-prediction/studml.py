#perfecto
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit.components.v1 as components

# --- Login Credentials ---
USERNAME = "admin"
PASSWORD = "1234"

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
def login():
    st.markdown("<h2 style='color:white;'>🔐 Login</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        st.write("username = admin   pass = 1234")

        if submit:
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
            else:
                st.error("Invalid username or password")

# --- Check Login ---
if not st.session_state.logged_in:
    login()
    st.stop()

# --- Custom CSS Injection ---
st.markdown("""
<style>
    :root {
        --space-cadet: #2B2D42;
        --manatee: #8D99AE;
        --cultured: #EDF2F4;
        --imperial-red: #EF233C;
        --persian-green: #2A9D8F;
    }
    .stApp {
        background: var(--space-cadet) !important;
        color: var(--cultured) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stNumberInput, .stSelectbox, .stSlider, .stTextInput {
        background: rgba(237, 242, 244, 0.05) !important;
        border: 1px solid var(--manatee) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    .stButton>button {
        background: var(--persian-green) !important;
        border: none !important;
        color: var(--cultured) !important;
        border-radius: 8px !important;
        padding: 14px 32px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(42, 157, 143, 0.3) !important;
    }
    .metric-card {
        background: rgba(237, 242, 244, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(237, 242, 244, 0.1);
    }
    .dashboard-container {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    model = joblib.load('student_model.pkl')
except Exception as e:
    st.error(f"Model loading error: {str(e)}")
    st.stop()

# --- Header Section ---
components.html("""
<div class="dashboard-container">
    <h1 class="header-title" style="color: white;">Academic Performance Predictor</h1>
    <p class="header-subtitle" style="color: white;">Data-Driven Student Success Analysis</p>
</div>
""")

# --- Input Section ---
with st.form("student_form"):
    student_name = st.text_input("Student Name")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### Student Profile")
        academic_history = st.number_input("Academic History (GPA)", 0.0, 4.0, 3.2)
        previous_scores = st.number_input("Previous Scores (%)", 0.0, 100.0, 78.0)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 88)
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        parental_support = st.selectbox("Parental Support Level", ["Low", "Medium", "High"])
        parental_education = st.selectbox("Parental Education", ["High School", "Bachelor's", "Master's", "PhD"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        st.markdown("### Behavioral Metrics")
        mental_health = st.slider("Mental Health Score (1-10)", 1, 10, 7)
        study_time = st.slider("Weekly Study Hours", 0, 40, 18)
        sleep_hours = st.slider("Daily Sleep Hours", 0, 12, 7)
        physical_activity = st.slider("Physical Activity Hours", 0, 20, 4)
        extracurricular = st.selectbox("Extracurricular Activities", ["None", "Low", "High"])
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        mental_illness = st.checkbox("History of Mental Illness")

    with st.expander("🔍 Advanced Parameters"):
        col3, col4 = st.columns(2)
        with col3:
            understand_ability = st.slider("Understanding Ability", 1, 10, 7)
            school_type = st.selectbox("Institution Type", ["Public", "Private"])
            distance_home = st.slider("Distance from Home (km)", 0, 50, 5)

        with col4:
            family_income = st.selectbox("Family Income Bracket", ["Low", "Middle", "High"])
            internet_access = st.selectbox("Internet Access Quality", ["Limited", "Unrestricted"])
            resources = st.selectbox("Resource Accessibility", ["Limited", "Moderate", "Full"])

    submit_button = st.form_submit_button("Analyze Performance")

# --- Feature Mapping ---
mapping = {
    "Low": 0, "Medium": 1, "High": 2,
    "None": 0, "Limited": 0, "Unrestricted": 1,
    "Negative": 0, "Neutral": 1, "Positive": 2,
    "Male": 0, "Female": 1,
    "High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3,
    "Public": 0, "Private": 1,
    "Middle": 1
}
# --- Prediction Logic ---
if submit_button:
    try:
        features = np.zeros((1, 19))
        features[0] = [
            academic_history,
            attendance,
            mapping[parental_support],
            mapping[resources],
            mapping[extracurricular],
            mental_health,
            previous_scores,
            mapping[motivation],
            mapping[internet_access],
            study_time,
            mapping[family_income],
            understand_ability,
            mapping[school_type],
            mapping[peer_influence],
            physical_activity,
            1 if mental_illness else 0,
            mapping[parental_education],
            distance_home,
            mapping[gender]
        ]

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # --- Visualization Section ---
        st.markdown("## Analysis Results")
        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("### Success Probability")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2A9D8F"},
                    'steps': [
                        {'range': [0, 50], 'color': "#2B2D42"},
                        {'range': [50, 100], 'color': "#264653"}],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("### Key Metrics")
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8D99AE;">Risk Factors</p>
                <h3 style="color: #2A9D8F;">{(1-probability)*100:.1f}%</h3>
                <p>Predicted Failure Risk</p>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("### Recommendation")
            rec_text = "Increase academic support" if probability < 0.7 else "Maintain current strategy"
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8D99AE;">Suggested Action</p>
                <h3 style="color: #2A9D8F;">{rec_text}</h3>
                <p>Based on current profile</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("## 📚 Personalized Suggestions")
        if probability < 0.5:
            st.warning("**You’re in the low performance zone.**")
            st.markdown("""
            - Seek tutoring or academic coaching
            - Increase weekly study time and stick to a fixed routine
            - Focus on mental health and physical activity
            - Improve sleep schedule and reduce distractions
            - Engage with supportive peer groups
            - Talk with teachers and parents for regular feedback
            """)
        elif probability < 0.8:
            st.info("**Moderate score – room for improvement.**")
            st.markdown("""
            - Stay consistent with study habits
            - Join or form a study group
            - Balance academics with healthy lifestyle
            - Track weekly goals and progress
            - Practice self-assessment and revise weak topics
            """)
        else:
            st.success("**Great performance – keep it up!**")
            st.markdown("""
            - Maintain current strategies and avoid burnout
            - Help others – teaching is a great way to retain knowledge
            - Prepare ahead for advanced topics
            - Stay focused, avoid overconfidence
            - Continue healthy habits in sleep and activity
            """)

        # --- Prepare Data to Save ---
        input_data = {
            "Name":student_name,
            "Academic History": academic_history,
            "Previous Scores": previous_scores,
            "Attendance": attendance,
            "Motivation": motivation,
            "Parental Support": parental_support,
            "Parental Education": parental_education,
            "Gender": gender,
            "Mental Health": mental_health,
            "Study Time": study_time,
            "Sleep Hours": sleep_hours,
            "Physical Activity": physical_activity,
            "Extracurricular": extracurricular,
            "Peer Influence": peer_influence,
            "Mental Illness": "Yes" if mental_illness else "No",
            "Understanding Ability": understand_ability,
            "School Type": school_type,
            "Distance from Home": distance_home,
            "Family Income": family_income,
            "Internet Access": internet_access,
            "Resources": resources,
            "Predicted Probability": round(probability, 4),
            "Prediction": "Likely to Succeed" if prediction[0] == 1 else "At Risk"
        }

        # Create DataFrame from the input data
        df = pd.DataFrame([input_data])
        excel_path = "C:/AI-powered-student-academic-prediction/saveddata.xlsx"

        # --- Save Student Input to Excel ---
        try:
            # Try reading the existing Excel file to append new data
            existing_df = pd.read_excel(excel_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # If the file doesn't exist, it will be created when saving
            pass

    except Exception as e:
        # Handle any errors that occur during the prediction process
        st.error(f"❌ Error during prediction: {str(e)}")

    finally:
        # This block will always be executed
        try:
            # Save the combined DataFrame to the Excel file
            df.to_excel(excel_path, index=False)
            st.success("✅ Student data saved successfully Saved " + student_name)
        except Exception as e:
            # Handle any errors that occur during the saving process
            st.error(f"❌ Error saving to Excel: {str(e)}")
