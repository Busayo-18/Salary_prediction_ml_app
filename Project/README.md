# Employee Prediction Model

---

# Project Overview
This project is a Machine Learning web application that predicts employee salaries based on years of experience, years at company, department, education level, performance rating, and performance rating missing indicator.

The model was trained using a Voting Regressor that combines:
- Linear Regression
- Random Forest Regressor

The application is deployed using Streamlit.

---

# Live Demo
http://localhost:8501/

---

# Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

# Machine Learning Approach

1. Data Cleaning
2. Feature Engineering (including Performance Missing Indicator)
3. Train-Test Split
4. Voting Regressor Model
5. Model Evaluation (R² Score, MAE, RMSE)
6. Model Serialization using Joblib
7. Streamlit Deployment

---

# Model Performance

Target variable mean: 152093.0
R² Score: 0.92
MAE: 8347.962
RMSE: 11525.681

---

# Project Structure
Salary_prediction_ml_app
|__data
    |__ clean_salary_dataset.csv
|__notebook
    |__ salary_model.ipynb
|__pipeline.py
|__README.md
|__requirements.txt
|__salary_prediction_app.py
|__salary_prediction_pipeline.pkl

---

# How to run locally

Clone the repository:
git clone <your-repo-link>
cd salary-prediction-ml-app

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run Project/salar_prediction_app.py

---

# Author
Olagunju Amina  
Aspiring Data Scientist  