import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
### get file
file = pd.read_csv('Social_Network_Ads.csv')
##print(file.head())

### get age and estimated salary columns

x = file.iloc[:, 2:4].values
print('x-shape', x.shape)

# get Y
y = file.iloc[:, -1].values
print('y shape', y.shape)

from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)

print('trained x', x_train.shape)  ##should be 80% of dataset
print('tested x', x_test.shape)  ### 20% left

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
# print('scalar fit transformed', x_train)
x_test = scaler.transform(x_test)
##print('transformed test', x_test)

## apply knn to your data
## get K1 first method sqrt

k = np.sqrt(x_train.shape[0])
print('k=', math.floor(k))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=math.floor(k))

##model train
knn.fit(x_train, y_train)

plot_decision_regions(x, y, clf=knn, legend=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Knn with K='+ str(math.floor(k)))
plt.show()
y_pred = knn.predict(x_test)
##print('pred Y',y_pred)

print('shape y pred: ', y_pred.shape)
print('shape y test: ', y_test.shape)

print('first accuracy', accuracy_score(y_test, y_pred))
print('Variance Score', knn.score(x_train, y_train))
print('Mean Cross Validations score', np.mean(cross_val_score(knn, x, y)))
print('Mean square error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
confusionMatrix = confusion_matrix(y_test, y_pred)
print('confusion matrix', confusionMatrix)
#display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels={})

######## another way

accuracy = []
error_train = []
error_test = []
###comment
#for i in range(1, 26):
    #knn = KNeighborsClassifier(n_neighbors=i)
    #knn.fit(x_train, y_train)
    # xc = confusion_matrix(y_train, knn.predict(x_train))
    # yc = confusion_matrix(y_test, knn.predict(x_test))
    # error_train.append((x[0][1] + x[1][0])/x.sum())
    # error_test.append((y[0][1] + y[1][0])/y.sum())
    #accuracy.append(accuracy_score(y_test, knn.predict(x_test)))
#### Plot the errors
# plt.plot(range(1,26), error_train, label='Error rate at training')
# plt.plot(range(1,26), error_test, label='Error rate at test')
# plt.xlabel('K')
# plt.ylabel('Error')
# plt.show()

#print(len(accuracy))
#plt.plot(range(1, 26), accuracy)
#plt.show()
#print(accuracy, 'max value', max(accuracy))
#maxi = accuracy.index(max(accuracy))
#print('maxi', maxi)
# accuracy.index(max(accuracy))
#knn = KNeighborsClassifier(n_neighbors=maxi)
#knn.fit(x_train, y_train)
#y_pred = knn.predict(x_test)
#result = accuracy_score(y_test, y_pred)
#print('result', result)
#### end

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
########

def predict():
    age = int(input('Enter age: '))
    salary = int(input('Enter the salary: '))
    new_x = np.array([[age], [salary]]).reshape(1, 2)
    new_x = scaler.transform(new_x)
    if knn.predict(new_x)[0] == 0:
        print( 'Is not going to purchase')
    else:
        print('Is going to purchase')

predict()