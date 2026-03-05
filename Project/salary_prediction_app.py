# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
salary_model = joblib.load('Project/salary_prediction_pipeline.pkl')
print(f'\n Model loaded successfully \n')

#create title
st.title('Employee Salary Prediction App')

#create input fields for the user to enter the features
st.write('Please enter employee details to predict the monthly salary:')

Experience = st.number_input('YearsExperience')
Years_At_Company = st.number_input('YearsAtCompany')
Department = st.selectbox('Department', ['Sales', 'Engineering', 'HR', 'Marketing', 'Operations', 'Finance'])
Education_Level = st.selectbox('EducationLevel', ['High School', 'Bachelor', 'Master', 'PhD'])
Performance_Rating = st.slider('PerformanceRating', min_value=1, max_value=5, step=1)
Monthly_Hours_Worked = st.number_input('MonthlyHoursWorked')
Missing_Performance_Rating = st.selectbox('PerformanceRating_missing', [0, 1])

#create a button to make the prediction
if st.button('Predict Salary'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'YearsExperience': [Experience],
        'YearsAtCompany': [Years_At_Company],
        'Department': [Department],
        'EducationLevel': [Education_Level],
        'PerformanceRating': [Performance_Rating],
        'MonthlyHoursWorked': [Monthly_Hours_Worked],
        'PerformanceRating_missing': [Missing_Performance_Rating]
    })
    
    # Make the prediction using the loaded model
    predicted_salary = salary_model.predict(input_data)[0]
    
    # Display the predicted salary
    st.write(f'Predicted Monthly Salary: ${predicted_salary:.2f}')
