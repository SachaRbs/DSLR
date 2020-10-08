import sys
import pandas as pd

def _count(col):
    count = 0
    for i in col:
        if pd.isna(i) is False:
            count = count + 1
    return(count)

def get_min(col):
    col = col.sort_values()
    print(col)
    print(col[0])
    return col[0]

def _describe(df):
    df = df._get_numeric_data()
    name = []
    count = []
    for col in df:
        name.append(col)
        count.append(_count(df[col]))
        _min = get_min(df[col])
        # _max.append = get_max(df, col)


    print('----------------')
    print(name)
    print(count)
    print(_min)
    print('----------------')


def main():
    if len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1])
        print(df.describe())
        print('----------------------------------------')
        _describe(df)
    else:
        print("Usage: python describy.py data.csv")
if __name__ == "__main__":
    main()