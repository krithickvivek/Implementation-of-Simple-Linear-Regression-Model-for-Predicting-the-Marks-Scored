# Ex-No:2 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: Krithick Vivekananda.
#RegisterNumber:  212223240075
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


df=pd.read_csv("student_scores.csv")
print("Dataset :")
print(df)
print()
print("Head Values :\n",df.head())
print("Tail Values :\n",df.tail())
print()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print("X_Values :\n",X)
print()
print("Y_Values :\n",Y)
print()


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


print("Y_prediction Values : \n",Y_pred)
print("\nY_test Values : \n",Y_test)


print()

print("Training Set Graph :")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set) ")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


print()

print("Test Set Graph :")
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores (Test Set) ")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print()


print("Values for MSE, MAE, & RMSE :")
MSE=mean_squared_error(Y_test,Y_pred)
print("MSE = ",MSE)
MAE=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
```

## Output:
![Alt text](image.png)
![Alt text](image-1.png)
![Alt text](<Screenshot 2024-02-22 205013.png>)
![Alt text](<Screenshot 2024-02-22 205028.png>)

![Alt text](<Screenshot 2024-02-22 205045.png>)
![Alt text](<Screenshot 2024-02-22 205902.png>)
![Alt text](<Screenshot 2024-02-22 205923.png>)
![Alt text](<Screenshot 2024-02-22 205947.png>)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
