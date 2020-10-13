import sys
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df):
    """print the correlation matrix to find which features are more corralated
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    we have a correlation of -1 between Astronomy and Defense Against the Dark Arts"""

    plt.scatter(df['Astronomy'], df['Defense Against the Dark Arts'])
    plt.ylabel("Defense Against the Dark Arts")
    plt.xlabel('Astronomy')
    plt.legend()
    plt.savefig('../Images/scatter_plot.png')
    print('image saved in Image/scatter_plot.png')

def main():
    if len(sys.argv) == 2:
        try:
            df = pd.read_csv(sys.argv[1])
            scatter_plot(df)
        except:
            print("error")

if __name__ == "__main__":
    main()