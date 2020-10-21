import argparse
import numpy as np
import sys
import pandas as pd

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

def _describe(df):
    df = df._get_numeric_data()
    describe = []
    for col in df:
        describe.append(extract_value(col, np.array(df[col])))
    describe = pd.DataFrame(data=describe).set_index('name').T
    return describe
    print(describe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to data file')
    args = parser.parse_args()
    try:
        df = pd.read_csv(args.data)
        data = np.array(df._get_numeric_data())
        describe = _describe(df)
        print(describe)
    except:
        print('ERROR')
    

if __name__ == "__main__":
    main()
