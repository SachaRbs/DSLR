import argparse
import sys
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

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
        # ajouter une colonne de 1
        for i in range(0, self.nb_class):
            y_one = (y == i).astype(int)
            for _ in range(0, self.iterations):
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
        with open('../resources/theta.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.thetas)

def data_preprocessing(df):
    df = df.fillna(df.groupby('Hogwarts House').transform('mean'))
    df['Best Hand'] = np.where(df['Best Hand'] == 'Right', 0, 1)
    y = df['Hogwarts House']
    df = df.drop('Hogwarts House', axis=1)
    y = y.map(House)
    df_standardized_manual = (df - df.mean()) / df.std()
    X = np.array(df_standardized_manual)
    X = np.insert(X, 0, 1, axis=1)
    return X, y

# def print_metrics(matrix):
#     TP = np.diag(matrix)
#     FP = matrix.sum(axis=0) - np.diag(matrix)
#     FN = matrix.sum(axis=1) - np.diag(matrix)
#     TN = matrix.sum() - (FP + FN + TP)
#     FP = FP.astype(float)
#     FN = FN.astype(float)
#     TP = TP.astype(float)
#     TN = TN.astype(float)
#     # Sensitivity, hit rate, recall, or true positive rate
#     print("TPR = {}".format(TP/(TP+FN)))
#     # Specificity or true negative rate
#     print("TNR = {}".format(TN/(TN+FP)))
#     # Precision or positive predictive value
#     print("PPV = {}".format(TP/(TP+FP)))
#     # Negative predictive value
#     print("NPV = {}".format(TN/(TN+FN)))
#     # Fall out or false positive rate
#     print("FPR = {}".format(FP/(FP+TN)))
#     # False negative rate
#     print("FNR = {}".format(FN/(TP+FN)))
#     # False discovery rate
#     print("FDR = {}".format(FP/(TP+FP)))
#     # Overall accuracy for each class
#     print("acc = {}".format((TP+TN)/(TP+FP+FN+TN)))

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
        X, y = data_preprocessing(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

        model = MultiLogisticRegression(len(House))
        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        model.save_theta()
        if args.metrics:
            confusion_matrix(y, y_pred)
    except:
        print("error")

if __name__ == "__main__":
    main()
