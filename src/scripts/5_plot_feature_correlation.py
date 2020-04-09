import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main(path):
    data = pd.read_csv(path)
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Plots Feature Correlation Matrix')
    arg_parser.add_argument("-p", "--path", required=True)

    args = arg_parser.parse_args()
    main(args.path)
