import matplotlib.pyplot as plt
from tools.data import get_results
import seaborn as sns

def box(data_name):
    x = "Initials"
    y = "F1"
    hue = "Algorithm"
    data = get_results(data_name)
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x=x, y=y, data=data, saturation=100, width=.9)
    ax.set(xlabel="Algoritmos")
    plt.show()
    # plt.savefig("./results/images/boxplot")