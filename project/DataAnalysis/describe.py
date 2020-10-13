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

def describe_print(describe):
    for line in describe[0]:
        for categories in describe:
            if len(str(categories[line])) > categories["len"]:
                categories["len"] = len(str(categories[line]))
    
    for line in describe[0]:
        if line != "name":
            write = '{0:5s}'.format(line)
        else:
            write = '{0:5s}'.format("")
        for categories in describe:
            write = write + '{:>{width}s}'.format(str(categories[line]), width=categories["len"] + 2)
        if line != "len":
            print(write)

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
                 "max": 0,
                 "len":0}
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
    describe_print(describe)

def main():
    if len(sys.argv) == 2:
        try:
            df = pd.read_csv(sys.argv[1])
            data = np.array(df._get_numeric_data())
            _describe(df)
        except:
            print("ERROR")
    else:
        print("Usage: python describy.py data.csv")
if __name__ == "__main__":
    main()
