import numpy as np
import sys
import pandas as pd
import argparse

def _std(mean, count, data):
    if count != 0:
        diff = 0
        for value in data:
            if pd.isna(value) is False:
                diff = diff + (value - mean)**2
        std = (diff / count)**0.5
        return std
    return np.nan

def extract_value(name, data):
    _count = 0.0
    describe = {"name" : name,
                "count": 0,
                 "mean": 0,
                 "std": 0,
                 "min": 0,
                 "25%": 0,
                 "50%": 0,
                 "75%": 0,
                 "max": 0}
    for value in data:
        if pd.isna(value) is False:
            describe["count"] = describe["count"] + 1
            describe["mean"] = describe["mean"] + value
    if describe["count"] != 0:
        describe["mean"] = describe["mean"] / describe["count"]
    else:
        describe["mean"] = np.nan

    data.sort()
    quart = (describe["count"] - 1) / 4
    quart_i = int(quart)
    describe["min"] = format(data[0], '.6f')
    describe["25%"] = format(data[quart_i] + ((quart - quart_i) * (data[quart_i + 1] - data[quart_i])), '.6f')
    describe["50%"] = format(data[quart_i * 2] + (((quart * 2) - (quart_i * 2)) * (data[(quart_i * 2) + 1] - data[quart_i * 2])), '.6f')
    describe["75%"] = format(data[quart_i * 3] + (((quart * 3) - (quart_i * 3)) * (data[(quart_i * 3) + 1] - data[quart_i * 3])), '.6f')
    describe["max"] = format(data[describe["count"] - 1], '.6f')
    describe["std"] = format(_std(describe["mean"], describe["count"], data), '.6f')
    describe['mean'] = format(describe['mean'], '.6f')
    if describe["count"] != 0:
        describe['count'] = format(describe['count'], '.6f')

    return describe

def correlation_matrix(df, describe):
    try:
        df = df.drop('Hogwarts House', axis=1)
    except:
        pass
    df = df.fillna(df.mean())
    columns = df.columns
    matrix = {'_': columns}
    arr = np.array(df.T)
    m = len(arr[0])
    for i in range(len(arr)):
        matrix[columns[i]] = []
        for j in range(len(arr)):
            cov = sum((arr[i] - float(describe[columns[i]]['mean'])) * (arr[j] - float(describe[columns[j]]['mean']))) / m
            matrix[columns[i]].append(cov / (float(describe[columns[i]]['std']) * float(describe[columns[j]]['std'])))

    matrix = pd.DataFrame(matrix)
    matrix = matrix.set_index('_')
    print(matrix)

def _describe(df):
    describe = []
    for col in df:
        describe.append(extract_value(col, np.array(df[col])))
    describe = pd.DataFrame(data=describe).set_index('name').T
    print(describe)
    return describe

def main():
    # try:
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to data file')
    parser.add_argument('-c', '--correlation_matrix', help='print the correlation matrix of the dataFrame', action='store_true')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    df = df._get_numeric_data()
    describe = _describe(df)
    print(df.describe())
    if args.correlation_matrix:
        correlation_matrix(df, describe)
    # except:
        # print("ERROR")
    

if __name__ == "__main__":
    main()
