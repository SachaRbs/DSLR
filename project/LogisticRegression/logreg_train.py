import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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
        self.thetas = np.zeros((self.nb_class, X.shape[1]))
        m = len(y)
        # ajouter une colonne de 1
        for i in range(0, self.nb_class):
            y_one = (y == i).astype(int)
            for _ in range(0, self.iterations):
                z = X.dot(self.thetas[i])
                h = self.sigmoid(z)
                self.thetas[i] = self.gradient(X, h, self.thetas[i], y_one, m)
        # print(self.thetas)
        
    def gradient(self, X, h, theta, y, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def predict(self, X):
        #format predict
        for i in range(0, self.nb_class):
            print(self.sigmoid(X.dot(self.thetas[i])))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_pred):
        pass

def data_preprocessing(df):
    df['Best Hand'] = np.where(df['Best Hand'] == 'Right', 0, 1)
    X = df.drop('Hogwarts House', axis=1)
    y = df['Hogwarts House']
    y = y.map(House)
    return np.array(X), np.array(y), House

def main():
    if len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1])
        df = df.drop(['First Name', 'Last Name', 'Birthday', 'Index'],axis=1)
        df = df.dropna()
        # print(df.isnull().values.any())
        X, y, House = data_preprocessing(df)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = MultiLogisticRegression(len(House))
        model.fit(X, y)
        model.predict(X)
        # print(list(House.keys())[list(House.values()).index(0)])

if __name__ == "__main__":
    main()