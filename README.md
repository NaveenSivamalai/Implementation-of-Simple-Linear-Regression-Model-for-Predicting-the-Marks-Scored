# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.
2. Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure. 
3. Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
4. Train Linear Regression Model: Initialize and train a Linear Regression model using the training data.
5.Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEEN S
RegisterNumber: 212222110030  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("student_scores.csv") 
df.head()
df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orangered",s=60)
plt.plot(x_train,regressor.predict(x_train),color="darkviolet",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()

plt.scatter(x_test,y_test,color="seagreen",s=60)
plt.plot(x_test,regressor.predict(x_test),color="cyan",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()


mse=mean_squared_error(_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```
# Output:
## Head:

![263004737-18f3af86-9ae2-4494-bb61-636b83a7bcd5](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/ce5a3a8c-9699-4cbf-b00f-3c6c96c3bfc5)

## Tail:

![263004822-0a120341-9b3f-4ee2-8740-1ea5cdc71610](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/75994a54-f77b-4356-9455-8849a4bec753)

## Array value of X:

![263005208-e296242a-26a5-4ba3-9736-86bc8fe4e85c](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/a32352dc-8688-4c44-9322-b2cec071d1c3)

## Array value of Y:
![263005383-c25d4900-5e51-4bf7-8db7-1ec80106ab55](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/21e9d326-1481-4f79-93dd-212387a62406)


## Values of Y prediction:
![263005811-611f79c2-9b85-47f4-9a9f-6fbe4dc4e7ee](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/d9db1aa4-03d6-48c0-84ac-1e5c5c351988)


## Array values of Y test:
![263006234-df0a81e5-6894-4e45-9037-6dce6cb3a7d7](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/0e93a00e-ccff-4884-92ca-9a5ed2a8dfbd)

## Training Set Graph:
![263006460-42c59aa0-503e-4dc8-b41a-315bdd9680c1](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/916d35bd-332f-4524-8b5e-bbfa87506e9a)

## Test Set Graph:
![263007034-8e868618-b584-4339-a91e-3b07e48301ba](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/bcd184fa-b55c-41f1-88e7-c8b66faea126)

## Values of MSE, MAE and RMSE:

![263006689-08d3470e-d71f-43e9-971f-e4871b589271](https://github.com/NaveenSivamalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123792574/ab98351e-3bfb-4037-a7cf-49341e529f2f)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
