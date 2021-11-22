import numpy as np
# Imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogisticRegression(l1_ratio=0.0001, max_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


print("LR classification accuracy:", accuracy(y_test, predictions))
def query():
    data = int(input('Enter data to know: '))
    if predictions[data] == 0:
        print('malignant')
    else:
        print('benign')


query()