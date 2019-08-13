import numpy as np
import pandas as pd
import sklearn

dataset = pd.read_csv('Salary_Data.csv')


X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print(Y_pred)

print(model.predict(np.array([[15]])))





