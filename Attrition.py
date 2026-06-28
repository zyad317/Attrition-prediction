import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import json

model = pickle.load(open("model.pkl","rb"))
transformation =pickle.load(open("power_transformation.pkl","rb"))
encoder = pickle.load(open("encoding.pkl", "rb"))

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


st.set_page_config(
    page_title = "streamlit prediction App"
)
st.title("Attrition prediction App")
st.write("this app for predict if there Attrition or not")



education_map = {
    "Below College": 1,
    "College": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctor": 5
}

environment_satisfaction_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

job_involvement_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

job_level_map = {
    "Entry Level": 1,
    "Intermediate Level": 2,
    "Senior Level": 3,
    "Managerial Level": 4,
    "Executive Level": 5
}

job_satisfaction_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

performance_rating_map = {
    "Good": 3,
    "Excellent": 4
}

relationship_satisfaction_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

stock_option_level_map = {
    "No Stock Options": 0,
    "Low Level": 1,
    "Medium Level": 2,
    "High Level": 3
}

work_life_balance_map = {
    "Bad": 1,
    "Good": 2,
    "Better": 3,
    "Best": 4
}






business_travel_map = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}

department_map = {
    "Sales": 0,
    "Research & Development": 1,
    "Human Resources": 2
}
 
education_field_map = {
    "Life Sciences": 0,
    "Medical": 1,
    "Marketing": 2,
    "Technical Degree": 3,
    "Human Resources": 4,
    "Other": 5
}

gender_map = {
    "Male": 0,
    "Female": 1
}

job_role_map = {
    "Sales Executive": 0,
    "Research Scientist": 1,
    "Laboratory Technician": 2,
    "Manufacturing Director": 3,
    "Healthcare Representative": 4,
    "Manager": 5,
    "Sales Representative": 6,
    "Research Director": 7,
    "Human Resources": 8
}

marital_status_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

overtime_map = {
    "No": 0,
    "Yes": 1
}


business_travel = st.selectbox("Business Travel", list(business_travel_map.keys()))
business_travel_num = business_travel_map[business_travel]

department = st.selectbox("Department", list(department_map.keys()))
department_num = department_map[department]

education_field = st.selectbox("Education Field", list(education_field_map.keys()))
education_field_num = education_field_map[education_field]

gender = st.selectbox("Gender", list(gender_map.keys()))
gender_num = gender_map[gender]

job_role = st.selectbox("Job Role", list(job_role_map.keys()))
job_role_num = job_role_map[job_role]

marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
marital_status_num = marital_status_map[marital_status]

overtime = st.selectbox("OverTime", list(overtime_map.keys()))
overtime_num = overtime_map[overtime]


age = st.number_input(
    "What is your age?",
    min_value=18, max_value=60, value=36
)

daily_rate = st.number_input(
    "What is your daily rate?",
    min_value=100, max_value=1500, value=800
)

distance_from_home = st.number_input(
    "How far do you live from your workplace (km)?",
    min_value=1, max_value=29, value=10
)

education = st.selectbox(
    "What is your highest level of education?",
    list(education_map.keys())
)
education_num = education_map[education]

environment_satisfaction = st.selectbox(
    "How satisfied are you with your work environment?",
    list(environment_satisfaction_map.keys())
)
environment_satisfaction_num = environment_satisfaction_map[environment_satisfaction]

hourly_rate = st.number_input(
    "What is your hourly rate?",
    min_value=30, max_value=100, value=65
)

job_involvement = st.selectbox(
    "How involved are you in your job?",
    list(job_involvement_map.keys())
)
job_involvement_num = job_involvement_map[job_involvement]

job_level = st.selectbox(
    "What is your current job level?",
    list(job_level_map.keys())
)
job_level_num = job_level_map[job_level]

job_satisfaction = st.selectbox(
    "How satisfied are you with your current job?",
    list(job_satisfaction_map.keys())
)
job_satisfaction_num = job_satisfaction_map[job_satisfaction]

monthly_income = st.number_input(
    "What is your monthly income?",
    min_value=1000, max_value=20000, value=5000
)

monthly_rate = st.number_input(
    "What is your monthly rate?",
    min_value=2000, max_value=27000, value=12000
)

num_companies_worked = st.number_input(
    "How many companies have you worked for before this one?",
    min_value=0, max_value=9, value=2
)

percent_salary_hike = st.number_input(
    "What was your last salary increase (%)?",
    min_value=11, max_value=25, value=15
)

performance_rating = st.selectbox(
    "What is your latest performance rating?",
    list(performance_rating_map.keys())
)
performance_rating_num = performance_rating_map[performance_rating]

relationship_satisfaction = st.selectbox(
    "How satisfied are you with your workplace relationships?",
    list(relationship_satisfaction_map.keys())
)
relationship_satisfaction_num = relationship_satisfaction_map[relationship_satisfaction]

stock_option_level = st.selectbox(
    "What is your stock option level?",
    list(stock_option_level_map.keys())
)
stock_option_level_num = stock_option_level_map[stock_option_level]

total_working_years = st.number_input(
    "How many total years of work experience do you have?",
    min_value=0, max_value=40, value=10
)

training_times_last_year = st.number_input(
    "How many training programs did you attend last year?",
    min_value=0, max_value=6, value=3
)

work_life_balance = st.selectbox(
    "How would you rate your work-life balance?",
    list(work_life_balance_map.keys())
)
work_life_balance_num = work_life_balance_map[work_life_balance]

years_at_company = st.number_input(
    "How many years have you worked at this company?",
    min_value=0, max_value=40, value=5
)

years_in_current_role = st.number_input(
    "How many years have you been in your current role?",
    min_value=0, max_value=18, value=3
)

years_since_last_promotion = st.number_input(
    "How many years have passed since your last promotion?",
    min_value=0, max_value=15, value=1
)

years_with_curr_manager = st.number_input(
    "How many years have you worked with your current manager?",
    min_value=0, max_value=17, value=3
)

input_data = [
    business_travel_num,
    department_num,
    education_field_num,
    gender_num,
    job_role_num,
    marital_status_num,
    overtime_num,
    age,
    daily_rate,
    distance_from_home,
    education,
    education_num,
    environment_satisfaction,
    environment_satisfaction_num,
    hourly_rate,
    job_involvement,
    job_involvement_num,
    job_level ,
    job_level_num,
    job_satisfaction,
    job_satisfaction_num,
    monthly_income,
    monthly_rate,
    num_companies_worked,
    percent_salary_hike,
    performance_rating,
    performance_rating_num,
    relationship_satisfaction,
    relationship_satisfaction_num,
    stock_option_level,
    stock_option_level_num,
    total_working_years,
    training_times_last_year,
    work_life_balance,
    work_life_balance_num,
    years_at_company,
    years_in_current_role,
    years_since_last_promotion,
    years_with_curr_manager
    ]

if st.button("predict"):

    catigorical_input = pd.DataFrame([[business_travel, department, education_field,gender, job_role, marital_status, overtime]],
                              columns=['BusinessTravel', 'Department', 'EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime'])

    encoded = encoder.transform(catigorical_input)  
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    num_input = pd.DataFrame([[
                age,
                daily_rate,
                distance_from_home,
                education_num,
                environment_satisfaction_num,
                hourly_rate,
                job_involvement_num,
                job_level_num,
                job_satisfaction_num,
                monthly_income,
                monthly_rate,
                num_companies_worked,
                percent_salary_hike,
                performance_rating_num,
                relationship_satisfaction_num,
                stock_option_level_num,
                total_working_years,
                training_times_last_year,
                work_life_balance_num,
                years_at_company,
                years_in_current_role,
                years_since_last_promotion,
                years_with_curr_manager
]],
columns=[
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
])
    num_input[num_input.columns] = transformation.transform(num_input)

    
    final_input = pd.concat([num_input, encoded_df], axis=1)
    final_input = final_input[feature_names]  

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    if prediction[0] == 1:
        st.error(f" This employee is likely to LEAVE (Risk: {probability:.1%})")
    else:
        st.success(f" This employee is likely to STAY (Risk: {probability:.1%})")
