from packages.utils import *
from pprint import pprint

if __name__=="__main__":
    # is convenient to implement the label_dict inverse? key=idx, value=class_name 
    D, L, label_dict = load("data/trainData.txt")
    """ scatter_hist_per_feat(D, L, label_dict)
    plt.savefig("./plots/scatter") """

    print(var(D))
    print(mean(D))
    