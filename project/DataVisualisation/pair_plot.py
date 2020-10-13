import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pair_plot(df):
    sns_plot = sns.pairplot(df, hue="Hogwarts House")
    sns_plot.savefig("../Images/pair_plot.png")
    print('Image saved in Image/pair_plot.png')

def main():
    if len(sys.argv) == 2:
        # try:
        df = pd.read_csv(sys.argv[1])
        pair_plot(df)
        # except:
        #     print("error")

if __name__ == "__main__":
    main()