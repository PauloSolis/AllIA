import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN
data = pd.read_csv('Social_Network_Ads.csv')
#print(data.head)

X = data.iloc[:,2:4].values
y = data.iloc[:, -1].values

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### train model object of knn
knn = KNN(k=11)
knn.fit(X_train, y_train)
##knn.predict(np.array([60, 100000]).reshape(1,2))

def prediction():
    age=int(input('Enter the age'))
    salary = int(input('Enter the salary'))

    new_x = np.array([[age], [salary]]).reshape(1, 2)
    new_x = scaler.transform(new_x)

    result= knn.predict(new_x)

    if result == 0:
        print('Is not going to purchase')
    else:
        print('It will purchase')

prediction()