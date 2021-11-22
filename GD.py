class LR:
    ### bias and model are initialized
    def __init__(self):
        self.m = None
        self.b = None


    def fit(self, x_train, y_train):
        num = 0
        den = 0
        ## the gradient descend formula is applied
        for i in range(x_train.shape[0]):
            num = num + ((x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((x_train[i] - x_train.mean()) * ((x_train[i] - x_train.mean())))

        # weight and bias are updated
        self.m = num / den
        self.b = y_train.mean() - (self.m * x_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, x_test):
        print(x_test)
        return self.m * x_test + self.b


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('placement.csv')
# print(df.head())
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()
x = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_train.shape

lr = LR()
lr.fit(x_train, y_train)
# lr.predict(x_test[0])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = lr.predict(x_test)
print(y_test)
print(y_pred)
plt.scatter(df['cgpa'],df['package'])
plt.plot(x_train,lr.predict(x_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()
print("Mean absolute errror is", mean_absolute_error(y_test, y_pred))
print("Mean squared errror", mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))

print(x_test.shape)
def query():
    cpga = float(input('Enter cpga: '))
    print(lr.predict([[cpga]]))

query()