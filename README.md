# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Set Up Google Colab**: Open Google Colab and create a new notebook.
2. **Import Libraries**: Import necessary libraries like `numpy`, `pandas`, `scikit-learn`, and `matplotlib`.
3. **Load Dataset**: Upload the dataset (e.g., CSV file) and load it into a DataFrame.
4. **Explore the Dataset**: Display the first few rows and summary statistics.
5. **Visualize the Data**: Create a scatter plot to visualize the relationship between variables.
6. **Prepare Data for Training**: Split the dataset into features (X) and target (y), then into training and testing sets.
7. **Train the Linear Regression Model**: Instantiate and fit the `LinearRegression` model on the training data.
8. **Make Predictions**: Use the trained model to predict outcomes on the test data.
9. **Evaluate the Model**: Calculate evaluation metrics like Mean Squared Error (MSE) and R-squared.
10. **Visualize Predictions**: Plot the regression line against actual test data.
11. **Predict for New Data**: Use the model to predict for new input data.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NATARAJ KUMARAN S
RegisterNumber:  212223230137
*/
```
### Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
ds=pd.read_csv("/content/drive/MyDrive/student_scores.csv")
print(ds.shape)
print(ds.head())
ds.tail()
ds.describe()
ds.info()
x=ds.iloc[:,:-1].values
print(x)
y=ds.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_train,y_train)
y_pred=r.predict(x_test)
print(y_pred)
print(y_test)
```
## Output
![Screenshot 2024-08-28 103917](https://github.com/user-attachments/assets/7704ffe6-0148-4847-a05c-0ca19ab482ee)
![Screenshot 2024-08-28 103928](https://github.com/user-attachments/assets/76fe0b20-146d-4e60-a7fc-5b327f151327)


### matplotlib
```python
plt.scatter(x_train,y_train,color='pink')
plt.plot(x_train,r.predict(x_train),color='cyan')
plt.title('Hours vs Score')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()
```
## Output
![Screenshot 2024-08-28 105131](https://github.com/user-attachments/assets/c43236b5-b2f8-4784-a262-98dbfbb1ff61)

### MAE,MSE,RMSE
```python
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
```
## Output:
![Screenshot 2024-08-28 103942](https://github.com/user-attachments/assets/3ea4c803-15ca-4247-8be2-fc0fe3fab289)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
