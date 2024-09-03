from modules.data.dataset import split_db_in_folds

import numpy as np

def kfold_cross_validation(D, L, model_constructor, kwargs, k = 10):
    # TODO: find a way to evaluate the result (maybe an array of performance metrics? or return just the results?)  
    # TODO: on the labs, professor says something about shuffle on cross validation (use a different shuffle from split??)
    folds = split_db_in_folds(D, L, k)
    results = []
    for fold in range(len(folds)):
        DVAL, LVAL = folds[fold]

        folds_t =  folds[:fold] + folds[fold+1:]
        DTR = np.concatenate([samples for samples, _ in folds_t], axis=1)
        LTR = np.concatenate([labels for _, labels in folds_t])
        
        model = model_constructor(DTR, LTR, **kwargs)
        model.fit()
        predictions, l_scores = model.predict(DVAL)
        results.append((predictions, l_scores, LVAL))

    return results
