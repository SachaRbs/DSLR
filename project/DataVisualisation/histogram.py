import matplotlib.pyplot as pyplot
import pandas as pd
import sys

def historgam(df):
    course = df.std().idxmin()
    houses = df['Hogwarts House'].unique()
    group = df.groupby('Hogwarts House')
    data = {}
    for house in houses:
        data[house] = group.get_group(house)[course]
    # print(data)
    for i in data:
        pyplot.hist(data[i], alpha=0.25, label=i)
    pyplot.ylabel("nombre d'eleves")
    pyplot.xlabel('notes')
    pyplot.title(course)
    pyplot.legend()
    pyplot.show()

def main():
    if len(sys.argv) == 2:
        try:
            df = pd.read_csv(sys.argv[1])
            historgam(df)
        except:
            print("error")

if __name__ == "__main__":
    main()