def error_rate(L, predictions):
    wrong_p = (L!=predictions).sum()
    error_rate = wrong_p/L.size
    return error_rate

def accuracy(L, predictions):
    return 1 - error_rate(L, predictions)

