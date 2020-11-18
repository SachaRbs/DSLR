import argparse
import sys
import pandas as pd
import numpy as np
import csv
from progress.bar import Bar

House = {'Ravenclaw': 0,
         'Slytherin': 1,
         'Gryffindor': 2,
         'Hufflepuff': 3}

class MultiLogisticRegression():
    def __init__(self, nb_class, alpha=0.001, iterations=10000):
        self.alpha = alpha
        self.iterations = iterations
        self.nb_class = nb_class
    
    def fit(self, X, y):
        self.classe = np.unique(y).tolist()
        self.thetas = np.zeros((self.nb_class, X.shape[1]))
        m = len(y)
        for i in range(0, self.nb_class):
            y_one = (y == i).astype(int)
            with Bar("Logistic Regression : {}/{}".format(i, self.nb_class), max=self.iterations) as bar:
                for _ in range(0, self.iterations):
                    bar.next()
                    z = X.dot(self.thetas[i])
                    h = self.sigmoid(z)
                    self.thetas[i] = self.gradient(X, h, self.thetas[i], y_one, m)
        
    def gradient(self, X, h, theta, y, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def predict(self, X):
        X_predicted = [max((self.sigmoid(i.dot(self.thetas[c])), c) for c in range(0, len(self.thetas)))[1] for i in X ]
        return np.array(X_predicted)

    def score(self, X, y):
        score = sum(self.predict(X) == y) / len(y)
        return score

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def save_theta(self):
        print("saving weights...")
        with open('../resources/theta.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.thetas)

def data_preprocessing(df, train):
    y = None
    if train:
        df = df.fillna(df.groupby('Hogwarts House').transform('mean'))
        y = df['Hogwarts House']
        y = y.map(House)
    df['Best Hand'] = np.where(df['Best Hand'] == 'Right', 0, 1)
    df = df.drop('Hogwarts House', axis=1)
    df_standardized_manual = (df - df.mean()) / df.std()
    X = np.array(df_standardized_manual)
    X = np.insert(X, 0, 1, axis=1)
    return X, y

def confusion_matrix(y, y_pred):
    labels = [0, 1, 2, 3]
    matrix = pd.DataFrame(np.zeros((4, 4)))
    for label1 in labels:
        for label2 in labels:
            matrix[label1][label2] = ((y_pred == label1) & (y == label2)).sum()
    acc = (y_pred == y).sum() / len(y_pred)
    print("accuracy : {}".format(acc))
    print("-------Confusion Matrix------")
    print(matrix)

def main():
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("data", help="path to the data file")
        parser.add_argument("-m", '--metrics', help="print metrics of the model", action="store_true")

        args = parser.parse_args()

        df = pd.read_csv(args.data)
        df = df.drop(['First Name', 'Last Name', 'Birthday', 'Index'],axis=1)
        X, y = data_preprocessing(df, 1)

        model = MultiLogisticRegression(len(House))
        model.fit(X, y)
        y_pred = model.predict(X)
        print("Accuracy : {}".format(model.score(X, y)))
        model.save_theta()
        if args.metrics:
            confusion_matrix(y, y_pred)
    except:
        print("error")

if __name__ == "__main__":
    main()
