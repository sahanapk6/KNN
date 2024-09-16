
import numpy as np
import pandas as pd
import sklearn as sm
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import deque
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    x1 = np.asfarray(x1, float)
    x2 = np.asfarray(x2, float)
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, X_test, y_train, k):
    prediction = []
    for p1 in X_test:
        distances = []
        for p2 in X_train:
            distance = euclidean_distance(p1, p2)
            distances.append(distance)

        df_distance = pd.DataFrame(data=distances, columns=['distance'],
                                   index=y_train.index)

        # Sort distances, and only consider the k closest points
        df_nn = df_distance.sort_values(by=['distance'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        mostcommonlabel = counter.most_common()[0][0]
        prediction.append(mostcommonlabel)

    return prediction


def accuracy_k_knn_plotgraph(X, y, normalize):
    num_runs = 20
    k_values = list(range(1, 52, 2))

    list_of_accuracy_train = []
    list_of_accuracy_test = []
    for i in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.20,
                                                            random_state=i,
                                                            shuffle=True)
        if normalize == 1:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        accuracy_train = []
        accuracy_test = []
        for k in k_values:
            prediction_train = knn_predict(X_train, X_train, y_train, k)
            accuracy_train.append(accuracy_score(y_train, prediction_train))

            prediction_test = knn_predict(X_train, X_test, y_train, k)
            accuracy_test.append(accuracy_score(y_test, prediction_test))

        list_of_accuracy_train.append(accuracy_train)
        list_of_accuracy_test.append(accuracy_test)

    train_data_accuracy = np.mean(np.array(list_of_accuracy_train), 0)
    test_data_accuracy = np.mean(np.array(list_of_accuracy_test), 0)

    train_std = np.std(np.array(list_of_accuracy_train), 0)
    test_std = np.std(np.array(list_of_accuracy_test), 0)

    # Plot the average accuracy of train and test data separately
    plt.errorbar(k_values, train_data_accuracy, xerr=None, yerr=train_std)
    plt.plot(k_values, train_data_accuracy)
    plt.title('Training Set Accuracy')
    plt.xlabel('k value')
    plt.ylabel('Accuracy percentage')
    plt.show()

    plt.errorbar(k_values, test_data_accuracy, xerr=None, yerr=test_std)
    plt.plot(k_values, test_data_accuracy)
    plt.title('Testing Set accuracy')
    plt.xlabel('k value')
    plt.ylabel('Accuracy percentage')
    plt.show()


# importing data
dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:, :-1]  # select all rows and all columns except the last one
y = dataset.iloc[:, -1]  # select all rows and only the last column

accuracy_k_knn_plotgraph(X, y, 1)

