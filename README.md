# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data: Read the dataset, extract features (X1) and target (y), and apply feature scaling using StandardScaler.<br>

2.Define Regression Function: Implement linear_regression() using gradient descent to update weights (theta) iteratively.<br>

3.Add Bias Term: Add a column of ones to X1 for the intercept term before training the model.<br>

4.Train the Model: Fit the model using scaled data to learn optimal theta values.<br>

5.Make Predictions: Scale new input data, predict using the trained model, and inverse-transform the output to original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SURIYA PRAKASH.S
RegisterNumber: 212223100055
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #Calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()

#Assuming rhe last column is your target variable 'y' and the preceding columns.
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)

scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target calue for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")

```

## Output:
![image](https://github.com/user-attachments/assets/083e4f38-c3dd-4f8f-8239-588ce65d9a49)
![image](https://github.com/user-attachments/assets/044dc783-66e2-48c1-8509-f13344575ad2)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
