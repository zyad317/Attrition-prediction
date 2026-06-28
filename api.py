from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json

app = FastAPI()

# Load model + preprocessing
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoding.pkl", "rb"))
transformer = pickle.load(open("power_transformation.pkl", "rb"))

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


# ---------------- INPUT SCHEMA ----------------
class EmployeeInput(BaseModel):
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    OverTime: str

    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict")
def predict(data: EmployeeInput):

    # categorical
    cat = pd.DataFrame([[
        data.BusinessTravel,
        data.Department,
        data.EducationField,
        data.Gender,
        data.JobRole,
        data.MaritalStatus,
        data.OverTime
    ]], columns=encoder.feature_names_in_)

    cat_encoded = encoder.transform(cat)
    cat_encoded = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out())

    # numeric
    num = pd.DataFrame([[
        data.Age, data.DailyRate, data.DistanceFromHome, data.Education,
        data.EnvironmentSatisfaction, data.HourlyRate,
        data.JobInvolvement, data.JobLevel, data.JobSatisfaction,
        data.MonthlyIncome, data.MonthlyRate, data.NumCompaniesWorked,
        data.PercentSalaryHike, data.PerformanceRating,
        data.RelationshipSatisfaction, data.StockOptionLevel,
        data.TotalWorkingYears, data.TrainingTimesLastYear,
        data.WorkLifeBalance, data.YearsAtCompany,
        data.YearsInCurrentRole, data.YearsSinceLastPromotion,
        data.YearsWithCurrManager
    ]], columns=[
        'Age','DailyRate','DistanceFromHome','Education',
        'EnvironmentSatisfaction','HourlyRate',
        'JobInvolvement','JobLevel','JobSatisfaction',
        'MonthlyIncome','MonthlyRate','NumCompaniesWorked',
        'PercentSalaryHike','PerformanceRating',
        'RelationshipSatisfaction','StockOptionLevel',
        'TotalWorkingYears','TrainingTimesLastYear',
        'WorkLifeBalance','YearsAtCompany',
        'YearsInCurrentRole','YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ])

    num = transformer.transform(num)

    final = pd.concat([num, cat_encoded], axis=1)
    final = final[feature_names]

    pred = model.predict(final)[0]
    prob = model.predict_proba(final)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
