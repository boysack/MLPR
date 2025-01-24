from modules.data.dataset import split_db_in_folds
from modules.models.model import Model
from modules.models.logistic_regression import LogisticRegression

import numpy as np

def kfold_scores_pooling(D, L, model_constructor: Model, kwargs, k = 10, seed = 0, preprocess_func = None, preprocess_args = []):
    # TODO: train using the model passed as argument, return all the pooled scores
    # TODO: on the labs, professor says something about shuffle on cross validation (use a different shuffle from split??)
    #folds, idx = split_db_in_folds(D, L, k, seed)
    # TODO: apply a method to limit the number of folds to the maximum of max(D.shape[1])
    if k < 2:
        raise Exception("You must have at least 2 fold to pool scores using kfold")
    if len(D.shape) == 1:
        D = D[np.newaxis, :]
    folds = split_db_in_folds(D, L, k, seed)
    l_scores = []
    predictions = []
    LVAL = []

    # code used for lab11 to check with professor results
    """ folds = [(D[:,::5], L[::5]),
             (D[:,1::5], L[1::5]),
             (D[:,2::5], L[2::5]),
             (D[:,3::5], L[3::5]),
             (D[:,4::5], L[4::5])] """

    for fold in range(len(folds)):
        KDVAL, KLVAL = folds[fold]
        # work since folds are simple lists (+ concatenate elementss)
        folds_t =  folds[:fold] + folds[fold+1:]
        DTR = np.concatenate([samples for samples, _ in folds_t], axis=1)
        LTR = np.concatenate([labels for _, labels in folds_t])

        if preprocess_func is not None:
            ret = preprocess_func(DTR, *preprocess_args)
            DTR = ret[0]
            KDVAL = preprocess_func(KDVAL, *ret[1:])[0]

        model = model_constructor(DTR, LTR, **kwargs)
        model.fit()
        k_predictions, k_l_scores = model.predict(KDVAL)

        l_scores.append(k_l_scores)
        predictions.append(k_predictions)
        LVAL.append(KLVAL)

    if len(l_scores[0].shape) > 1:
        l_scores = np.concatenate(l_scores, axis=1)
    else:
        l_scores = np.concatenate(l_scores)
    predictions = np.concatenate(predictions)
    LVAL = np.concatenate(LVAL)

    # return even LVAL, since shuffled
    return l_scores, predictions, LVAL #, idx

# TODO: label_dict? is llr, i.e. binary, I could use the true_idx to create a local label_dict
def llr_calibration(llr, L, label_dict, llr_val = None, LVAL = None, k = 10, l_priors = None):
    if len(llr.shape) != 1:
        raise Exception("You must pass log-likelihood ratios to perform calibration!")

    lr = LogisticRegression(llr, L, label_dict, l_priors=l_priors)
    lr.fit()

    if llr_val is not None and LVAL is not None:
        # single fold approach using manual splitted validation sets
        predictions, l_scores = lr.predict(llr_val)
    else:            
        l_scores, predictions, LVAL = kfold_scores_pooling(llr, L, LogisticRegression, {"label_dict":label_dict, "l_priors":l_priors}, k)

    return (l_scores, predictions, LVAL), lr

def model_fusion(scores, L, label_dict, scores_val = None, LVAL = None, k = 10, l_priors = None):
    lr = LogisticRegression(scores, L, label_dict, l_priors=l_priors)
    lr.fit()

    if scores_val is not None and LVAL is not None:
        # single fold approach using manual splitted validation sets
        predictions, pooled_scores = lr.predict(scores_val)
    else:            
        pooled_scores, predictions, LVAL = kfold_scores_pooling(scores, L, LogisticRegression, {"label_dict":label_dict, "l_priors":l_priors}, k)

    return (pooled_scores, predictions, LVAL), lr