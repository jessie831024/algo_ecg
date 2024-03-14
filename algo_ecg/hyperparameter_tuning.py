import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    train_test_split, GroupShuffleSplit, cross_val_score, GroupKFold, \
    StratifiedGroupKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from algo_ecg.data import import_physionet_data
from algo_ecg.preprocessing import preprocess_pid
from algo_ecg.feature_transformer import AllFeatureCustomTransformer, RemoveCorrelatedFeatures
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from timeout_decorator import timeout
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, \
    classification_report, roc_curve, confusion_matrix, auc, accuracy_score, precision_score

from joblib import parallel_backend
from itertools import product
from memory_profiler import profile

from xgboost import XGBClassifier
import pickle

from hyper_config import hyperparameter_options
import argparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
I have used the @timeout(1200) decorator for the run_search function, 
which will raise a TimeoutError if the function execution exceeds the specified timeout of 1200 seconds (20 minutes).
In the main function, I catch the TimeoutError and log an error message if it occurs. 
The script then continues execution or logs other exceptions if they occur.
'''
@timeout(6000)
@profile
def run_search(pipe, param_grid, groups, X_train, y_train, search_method, **kwargs):
    # Define search methods and their corresponding classes
    search_class = SEARCH_METHODS[search_method]
    search = search_class(pipe, param_grid, **kwargs)

    '''
    RandomizedSearchCV does not have the groups parameter in its fit method, while HalvingRandomSearchCV does.
    If the groups parameter is not included in the cross-validation process, it means that the cross-validation splits 
    may not properly account for grouping or clustering of data. This can lead to potential data leakage, 
    especially when dealing with datasets where samples are not independent, 
    and there is a need to ensure that samples from the same group or cluster are kept together in either the training 
    or testing sets.
    '''
    # Check if the search class supports 'groups' parameter
    if 'groups' in search.get_params().keys():
        # Only pass 'groups' to fit if the search class supports it
        scores = search.fit(X_train, y_train, groups=groups)
    else:
        # If 'groups' is not supported, fit without it
        scores = search.fit(X_train, y_train)
    return scores

def single_holdout (X, y, pids):
    slice_length = len(X[0])
    X = np.concatenate(X).reshape(-1, slice_length, 1)

    # Hold out a test set using
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    split = splitter.split(X, groups=pids)
    train_inds, test_inds = next(split)

    X_train = X[train_inds]
    X_test = X[test_inds]

    y_train = y[train_inds]
    y_test = y[test_inds]

    pids_train = pids[train_inds]
    pids_test = pids[test_inds]

    return X_train, X_test, y_train, y_test, pids_train, pids_test

def predict_perf(search, X_test, y_test):
    y_test_pred = search.predict(X_test)
    y_test_pred_prob = search.predict_proba(X_test)

    print(classification_report(y_test, y_test_pred))
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob[:, 1])
    print(auc(x=fpr, y=tpr))
    return auc

def main(pipe, param_grid, X_train, X_test, y_train, y_test, groups, search_method, logger, filename, **kwargs):
    try:
        scores = run_search(pipe, param_grid, groups, X_train, y_train, search_method, **kwargs)
        auc = predict_perf(scores, X_test, y_test)
        with open(filename, 'wb') as file:
            pickle.dump(scores, file)
            pickle.dump(auc, file)
    except TimeoutError:
        logger.error(f"Execution exceeded the timeout of {kwargs['timeout_value']} seconds.")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":

    SEARCH_METHODS = {
       'HalvingRandomSearchCV': HalvingRandomSearchCV,
       'GridSearchCV': GridSearchCV,
       'RandomizedSearchCV': RandomizedSearchCV
       # Add other search methods as needed
    }
    # Load data
    X_in, y, pids = import_physionet_data('/Users/jessie/data/PhysioNet_CinC', num_files_to_read=1000)

    # Preprocess data
    X, pids = preprocess_pid(X_in, pids)

    X_train, X_test, y_train, y_test, pids_train, pids_test = single_holdout(X, y, pids)

    # Convert to DataFrames
    X_train = pd.DataFrame.from_records(X_train)
    X_test = pd.DataFrame.from_records(X_test)

    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=['run_lr', 'run_xgb', 'run_svc'])
    args = parser.parse_args()

    my_pipe = hyperparameter_options[args.run]['my_pipe']
    my_param_grid = hyperparameter_options[args.run]['my_param_grid']
    my_search_method = hyperparameter_options[args.run]['search_method']
    additional_params = hyperparameter_options[args.run]['additional_params']

    filename = "search_{}_{}.pkl".format(args.run, time.strftime("%Y%m%d-%H%M%S"))

    # Specify the timeout value
    timeout_value = 6000
    # Run main with the specified pipeline and parameter grid
    main(my_pipe, my_param_grid, X_train, X_test, y_train, y_test, pids_train, my_search_method, logger, filename, \
         **additional_params)
