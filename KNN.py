import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print('test completed')

    def predict(self, X_test):
        y_pred = []
        for i in X_test:
            distance = []
            for j in self.X_train:
                distance.append(np.linalg.norm(i - j))

            # print(distance)
            n_neighbors = sorted(list(enumerate(distance)), key=lambda x: x[1])[0:self.k]
            label = self.classifier(n_neighbors)
            y_pred.append(label)
        return np.array(y_pred)

    def classifier(self, distance):
        label = []

        for i in distance:
            label.append(self.y_train[i[0]])
        return Counter(label).most_common()[0][0]
