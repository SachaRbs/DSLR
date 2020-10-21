import numpy as np
import csv
import argparse
import pandas as pd
from logreg_train import data_preprocessing
from sklearn.preprocessing import StandardScaler

House = {0: 'Ravenclaw',
         1: 'Slytherin',
         2: 'Gryffindor',
         3: 'Hufflepuff'}

def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def predict(X, thetas):
    y_pred = []
    for i in range(0, len(X)):
        y_pred.append([i, max((sigmoid(X[i].dot(thetas[c])), House[c]) for c in range(0, len(thetas)))[1]])
    return np.array(y_pred)

def print_csv(y_pred):
    with open('../resources/house.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([['Index', 'Hogwarts House']])
                writer.writerows(y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to data file')
    parser.add_argument('thetas', help='path to thetas file')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    thetas = np.genfromtxt(args.thetas, delimiter=',')
    df = df.drop(['First Name', 'Last Name', 'Birthday', 'Index'],axis=1)
    X, _, _ = data_preprocessing(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = predict(X, thetas)
    print_csv(y_pred)

if __name__ == "__main__":
    main()