# Import necessary libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor

# Load the dataset
salary_df = pd.read_csv('Project/data/clean_salary_dataset.csv')

#define features(X) and target variable(y)
X = salary_df.drop(columns=['MonthlySalary'])
y = salary_df['MonthlySalary']
print(f'Features and target variable defined successfully \n')
#display the first five rows of X
print(X.head())
#display the first five rows of y
print(y.head())
#drop employee id column and age column from X
X = X.drop(columns=['EmployeeID', 'Age'])

#split the columns of X into numerical and categorical columns
categorical_cols = ['Department', 'EducationLevel']
numerical_cols = [col for col in X.columns if col not in categorical_cols]
#print the categorical and numerical columns
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

#define the preprocessor for the categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

#define the models to be used in the pipeline
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=234)

#create a voting regressor using the defined models
voting_regressor = VotingRegressor(estimators=[('lr', lr), ('rf', rf)])

#create a pipeline that includes the preprocessor and the voting regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('voting_regressor', voting_regressor)
])
print(f'\n Pipeline created successfully \n')

#split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=234)

#train the model using the pipeline on the training data
pipeline.fit(X_train, y_train)

#make predictions on the test data
y_pred = pipeline.predict(X_test)

#evaluate the model using various metrics
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred),3))
print("Root Mean Squared Error:", round(root_mean_squared_error(y_test, y_pred),3))
print("R² Score:", round(r2_score(y_test, y_pred),3))
print('Target variable mean:', round(y.mean(),3))

#display the actual vs predicted values in a scatter plot
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')
plt.title("Pipeline: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show(block = 'false')
plt.pause(2)

#save the trained model using joblib
joblib.dump(pipeline, 'Project/salary_prediction_pipeline.pkl')
print(f'\n Model saved successfully')
print(f'model pipeline is ready for deployment \n')