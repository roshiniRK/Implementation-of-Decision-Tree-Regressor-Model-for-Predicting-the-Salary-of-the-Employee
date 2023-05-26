# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data
```
-Collect and clean data on employee salaries and features
-Split data into training and testing sets
```
2.Define your model
```
-Use a Decision Tree Regressor to recursively partition data based on input features
-Determine maximum depth of tree and other hyperparameters
```
3.Train your model
```
-Fit model to training data
-Calculate mean salary value for each subset
```
4.Evaluate your model
```
-Use model to make predictions on testing data
-Calculate metrics such as MAE and MSE to evaluate performance
```
5.Tune hyperparameters
```
-Experiment with different hyperparameters to improve performance
```
6.Deploy your model
```
Use model to make predictions on new data in real-world application.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ROSHINI R K
RegisterNumber:  212222230123
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### Initial dataset:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/b959cad1-b495-493a-840e-6daa72f6574b)
### Data Info:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/fc86bb86-aef9-4083-858e-c9156c2c0f6b)
### Optimization of null values:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/632ef939-cfae-404e-8d8b-6d7095518144)
### Converting string literals to numericl values using label encoder:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/3c2874cf-bb7c-4776-8a41-fb4c491a82b4)
### Assigning x and y values:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/f8aa3eea-a505-44f6-a74b-59665bd1f6c6)
### Mean Squared Error:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/1d8fb8bd-6829-4740-8eb8-575e682a0a58)
### R2 (variance):
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/6f8e1c44-3857-448f-b57c-b877a51e118d)
### Prediction:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/94e01ce8-0433-4678-bf85-365b2dc119a7)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

