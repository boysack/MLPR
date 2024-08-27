import setup
from modules.data.dataset import *
from labs.labs import lab02, lab03, lab04, lab05

if __name__=="__main__":

    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LAB 2 - features visualization

    #lab02(D, L, label_dict)
    
    # LAB 3 - pca, lda
    
    #lab03(D, L, label_dict, DTR, LTR, DVAL, LVAL)

    # LAB 4 - multivariate gaussian density

    #lab04(DTR, LTR, label_dict)

    # LAB 5 - multivariate gaussian classifiers

    #lab05(DTR, LTR, DVAL, LVAL, label_dict)

