import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import json

model = pickle.load(open("model.pkl","rb"))
transformation = pickle.load(open("power_transformation.pkl","rb"))
encoder = pickle.load(open("encoding.pkl", "rb"))

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


st.set_page_config(
    page_title="Attrition Prediction App",
    page_icon="📊",
    layout="wide"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

.stApp {
    background: #0B0F1A;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 4rem 2rem;
    max-width: 1100px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin: 2rem 0 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #818CF8;
    margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.15;
    margin: 0 0 0.75rem;
}
.hero h1 span {
    background: linear-gradient(90deg, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #94A3B8;
    font-size: 1rem;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}

/* ── Section cards ── */
.section-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
}
.section-label {
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    letter-spacing: 0.02em;
    margin-bottom: 0.25rem !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0B0F1A !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}

div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stNumberInput"] input:hover {
    border-color: #4F46E5 !important;
}

div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

/* Dropdown arrow & options */
div[data-testid="stSelectbox"] svg { color: #6366F1 !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: #78716C !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 1rem;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(120,113,108,0.35) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-stay {
    background: linear-gradient(135deg, #022C22, #064E3B);
    border: 1px solid #059669;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-leave {
    background: linear-gradient(135deg, #2D0A0A, #450A0A);
    border: 1px solid #DC2626;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.25rem 0;
}
.result-stay .result-title { color: #34D399; }
.result-leave .result-title { color: #FCA5A5; }
.result-sub { color: #94A3B8; font-size: 0.9rem; margin-top: 0.25rem; }
.risk-pill {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.result-stay .risk-pill { background: rgba(52,211,153,0.15); color: #34D399; }
.result-leave .risk-pill { background: rgba(252,165,165,0.15); color: #FCA5A5; }

/* ── Streamlit alerts override (hide default) ── */
.stAlert { display: none !important; }

/* ── Number input stepper buttons ── */
button[data-testid="stNumberInput-StepUp"],
button[data-testid="stNumberInput-StepDown"] {
    background: #1E293B !important;
    border-color: #1E293B !important;
    color: #818CF8 !important;
}

/* ── Columns gap ── */
div[data-testid="stHorizontalBlock"] { gap: 1.25rem; }
</style>
""", unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">HR Intelligence · Predictive Analytics</div>
    <h1>Employee <span>Attrition</span><br>Prediction</h1>
    <p>Fill in the employee profile below to assess flight risk. The model analyzes 30+ signals to predict whether this employee is likely to leave.</p>
</div>
""", unsafe_allow_html=True)


# ─── Maps ────────────────────────────────────────────────────────────────────
education_map = {"Below College":1,"College":2,"Bachelor":3,"Master":4,"Doctor":5}
environment_satisfaction_map = {"Low":1,"Medium":2,"High":3,"Very High":4}
job_involvement_map = {"Low":1,"Medium":2,"High":3,"Very High":4}
job_level_map = {"Entry Level":1,"Intermediate Level":2,"Senior Level":3,"Managerial Level":4,"Executive Level":5}
job_satisfaction_map = {"Low":1,"Medium":2,"High":3,"Very High":4}
performance_rating_map = {"Good":3,"Excellent":4}
relationship_satisfaction_map = {"Low":1,"Medium":2,"High":3,"Very High":4}
stock_option_level_map = {"No Stock Options":0,"Low Level":1,"Medium Level":2,"High Level":3}
work_life_balance_map = {"Bad":1,"Good":2,"Better":3,"Best":4}
business_travel_map = {"Non-Travel":0,"Travel_Rarely":1,"Travel_Frequently":2}
department_map = {"Sales":0,"Research & Development":1,"Human Resources":2}
education_field_map = {"Life Sciences":0,"Medical":1,"Marketing":2,"Technical Degree":3,"Human Resources":4,"Other":5}
gender_map = {"Male":0,"Female":1}
job_role_map = {"Sales Executive":0,"Research Scientist":1,"Laboratory Technician":2,"Manufacturing Director":3,"Healthcare Representative":4,"Manager":5,"Sales Representative":6,"Research Director":7,"Human Resources":8}
marital_status_map = {"Single":0,"Married":1,"Divorced":2}
overtime_map = {"No":0,"Yes":1}


# ─── Section 1: Identity ──────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-label">👤 &nbsp;Identity & Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=60, value=36)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    gender_num = gender_map[gender]
with c2:
    marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
    marital_status_num = marital_status_map[marital_status]
    education = st.selectbox("Education Level", list(education_map.keys()))
    education_num = education_map[education]
with c3:
    education_field = st.selectbox("Education Field", list(education_field_map.keys()))
    education_field_num = education_field_map[education_field]
    distance_from_home = st.number_input("Distance from Home (km)", min_value=1, max_value=29, value=10)
st.markdown('</div>', unsafe_allow_html=True)


# ─── Section 2: Role ─────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-label">💼 &nbsp;Role & Organization</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    department = st.selectbox("Department", list(department_map.keys()))
    department_num = department_map[department]
    job_role = st.selectbox("Job Role", list(job_role_map.keys()))
    job_role_num = job_role_map[job_role]
with c2:
    job_level = st.selectbox("Job Level", list(job_level_map.keys()))
    job_level_num = job_level_map[job_level]
    business_travel = st.selectbox("Business Travel", list(business_travel_map.keys()))
    business_travel_num = business_travel_map[business_travel]
with c3:
    overtime = st.selectbox("Overtime", list(overtime_map.keys()))
    overtime_num = overtime_map[overtime]
    stock_option_level = st.selectbox("Stock Option Level", list(stock_option_level_map.keys()))
    stock_option_level_num = stock_option_level_map[stock_option_level]
st.markdown('</div>', unsafe_allow_html=True)


# ─── Section 3: Compensation ──────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-label">💰 &nbsp;Compensation</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
with c2:
    daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800)
with c3:
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65)
with c4:
    monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=12000)

c1, c2 = st.columns(2)
with c1:
    percent_salary_hike = st.number_input("Last Salary Hike (%)", min_value=11, max_value=25, value=15)
with c2:
    performance_rating = st.selectbox("Performance Rating", list(performance_rating_map.keys()))
    performance_rating_num = performance_rating_map[performance_rating]
st.markdown('</div>', unsafe_allow_html=True)


# ─── Section 4: Experience ────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-label">📅 &nbsp;Experience & Tenure</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
with c2:
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=18, value=3)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
with c3:
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=17, value=3)
    num_companies_worked = st.number_input("Companies Worked Before", min_value=0, max_value=9, value=2)
st.markdown('</div>', unsafe_allow_html=True)


# ─── Section 5: Satisfaction ──────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-label">😊 &nbsp;Satisfaction & Engagement</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    job_satisfaction = st.selectbox("Job Satisfaction", list(job_satisfaction_map.keys()))
    job_satisfaction_num = job_satisfaction_map[job_satisfaction]
    environment_satisfaction = st.selectbox("Environment Satisfaction", list(environment_satisfaction_map.keys()))
    environment_satisfaction_num = environment_satisfaction_map[environment_satisfaction]
with c2:
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", list(relationship_satisfaction_map.keys()))
    relationship_satisfaction_num = relationship_satisfaction_map[relationship_satisfaction]
    work_life_balance = st.selectbox("Work-Life Balance", list(work_life_balance_map.keys()))
    work_life_balance_num = work_life_balance_map[work_life_balance]
with c3:
    job_involvement = st.selectbox("Job Involvement", list(job_involvement_map.keys()))
    job_involvement_num = job_involvement_map[job_involvement]
    training_times_last_year = st.number_input("Training Sessions Last Year", min_value=0, max_value=6, value=3)
st.markdown('</div>', unsafe_allow_html=True)


# ─── Predict ─────────────────────────────────────────────────────────────────
_, center, _ = st.columns([1, 2, 1])
with center:
    predict_clicked = st.button(" Run Attrition Analysis")

if predict_clicked:
    categorical_input = pd.DataFrame(
        [[business_travel, department, education_field, gender, job_role, marital_status, overtime]],
        columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    )
    encoded = encoder.transform(categorical_input)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    num_input = pd.DataFrame([[
        age, daily_rate, distance_from_home, education_num,
        environment_satisfaction_num, hourly_rate, job_involvement_num,
        job_level_num, job_satisfaction_num, monthly_income, monthly_rate,
        num_companies_worked, percent_salary_hike, performance_rating_num,
        relationship_satisfaction_num, stock_option_level_num,
        total_working_years, training_times_last_year, work_life_balance_num,
        years_at_company, years_in_current_role, years_since_last_promotion,
        years_with_curr_manager
    ]], columns=[
        'Age', 'DailyRate', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
        'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ])
    num_input[num_input.columns] = transformation.transform(num_input)

    final_input = pd.concat([num_input, encoded_df], axis=1)
    final_input = final_input[feature_names]

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-leave">
                <div class="result-icon">⚠️</div>
                <div class="result-title">High Flight Risk</div>
                <div class="result-sub">This employee is likely to leave the organization.</div>
                <div class="risk-pill">Attrition Risk: {probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-stay">
                <div class="result-icon">✅</div>
                <div class="result-title">Likely to Stay</div>
                <div class="result-sub">This employee shows strong retention signals.</div>
                <div class="risk-pill">Attrition Risk: {probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)