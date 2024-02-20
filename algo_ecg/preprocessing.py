import logging

import numpy as np

logger = logging.getLogger()

def check_bounds(x_in):
    return (x_in.min() < 0) and (x_in.max() > 0)

def remove_outliers(x_in, t=5):
    '''Remove values outside `t` standard deviations and replace them with
    linearly interpolated values.

    Arguments
    ---------
    x_in (np.ndararay)
        Input array.

    `t` (scalar)
        Number of standard deviations used to remove outliers.

    Returns
    -------
    ndarray, int
        The processed array and number of outliers found.

    '''
    outliers = np.abs(x_in - np.mean(x_in)) > t * np.std(x_in)
    if not np.sum(outliers):
        return x_in, 0
    x_in_nans = np.where(outliers, np.nan, x_in)
    x_in_interpolated = np.interp(
        np.arange(len(x_in)),
        np.where(~np.isnan(x_in_nans))[0],
        x_in_nans[~np.isnan(x_in_nans)]
    )
    logging.debug('')
    return x_in_interpolated, np.sum(outliers)

def flip_values(x_in, t=1.5):
    # Reflexion about y=0X
    if np.std(x_in[x_in < 0]) > t * np.std(x_in[x_in > 0]):
        return -x_in, True
    return x_in, False


def preprocess(X):
    X_proc = []
    for i, x_proc in enumerate(X):
        # logger.debug('Processing X[{}]'.format(i))
        if not check_bounds(x_proc):
            logger.debug('i={}: check bounds failed'.format(i))
        x_proc, num_outliers = remove_outliers(x_proc)
        if num_outliers:
            logger.debug('i={}: {} outliers found'.format(i, num_outliers))
        x_proc, flipped = flip_values(x_proc)
        if flipped:
            logger.debug('i={}: flipped values'.format(i))
        X_proc.append(x_proc)
    return X_proc


def preprocess_pid(X, pids):
    X_proc = []
    for i, x_proc in enumerate(X):
        # logger.debug('Processing X[{}]'.format(i))
        if not check_bounds(x_proc):
            logger.debug('i={}, pid={}: check bounds failed'.format(i, pids[i]))
        x_proc, num_outliers = remove_outliers(x_proc)
        if num_outliers:
            logger.debug('i={}, pid={}: {} outliers found'.format(i, pids[i], num_outliers))
        x_proc, flipped = flip_values(x_proc)
        if flipped:
            logger.debug('i={}, pid={}: flipped values'.format(i, pids[i]))
        X_proc.append(x_proc)
    return X_proc, pids

