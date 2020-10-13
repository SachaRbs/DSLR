import sys

class LogisticRegression():
    def __init__(self):
        pass

def main():
    if len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1])

if __name__ == "__main__":
    main()