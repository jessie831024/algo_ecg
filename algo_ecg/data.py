import glob
import logging
import pathlib
import pickle
import wget
import zipfile

import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger()
LABELS_URL = 'https://physionet.org/challenge/2017/REFERENCE-v3.csv'
TRAINING_URL = 'https://physionet.org/challenge/2017/training2017.zip'

def _make_path(path_str):
    return pathlib.Path(path_str).expanduser().absolute()

def download_physionet_data(output_dir):
    '''Download labels file ('REFERENCE-v3.csv') and training data
    ('training2017.zip') from Physionet webpage
    ('https://physionet.org/challenge/2017/#challenge-data)
    and unzip the training data.

    Parameters
    -----------

    output_dir (str)
        Directory where physionet data will be saved.

    '''
    data_path = _make_path(output_dir)
    if not data_path.exists():
        logging.info(
            'Directory {} does not exist and will be created.'.format(data_path)
        )
        pathlib.Path.mkdir(data_path, parents=True)

    labels_path = data_path / 'REFERENCE-v3.csv'
    wget.download(LABELS_URL, out=str(labels_path))
    logging.info('Labels file saved in {}.'.format(labels_path))

    training_data_path = data_path / 'training2017.zip'
    wget.download(TRAINING_URL, out=str(training_data_path))
    logging.info('Training data saved in {}.'.format(training_data_path))

    with zipfile.ZipFile(str(training_data_path), 'r') as zf:
        zf.extractall(str(data_path))
        logging.info(
            'Training data extracted under {}/training2017.'.format(data_path)
        )

def import_physionet_data(
    data_dir, labels_csv_file='REFERENCE-v3.csv',
    features_dir='training2017', num_files_to_read=None,
    train_ratio=0.7, binary=True, slice_length=9000
):
    '''Import physionet ECG data.

    Parameters
    -----------

    data_dir (str)
        Path containing physionet data.

    labels_csv_file (str)
        The name of the csv file containing data labels. This must be contained
        inside `data_dir_path`.

    features_dir (str)
        The sub-directory containing training data. This must be contained
        inside `data_dir_path`.

    num_files_to_read (int)
        How many training data files to read (the first `num_files_to_read` are
        read). Set to a low number (e.g. 100) when testing the code to speed up
        processing and training times.

    binary (bool)
        If True, then labels are assigned for binary classification (normal
        vs afib). Physionet provides additional labels 'other' and 'noise'
        which can be used for multi-class classification (not implemented yet).

    slice_length (int)
        Length of data slice. Data are sampled at 300Hz so e.g.
        slice_length=9000 corresponds to slices that are 30 seconds long.

    Returns
    -------

    X, y (np.darray, np.ndarray)

    Features and labels.
    '''
    data_dir_path = _make_path(data_dir)
    labels_index = {
        'N': 0,  # normal
        'A': 1,  # afib
    }
    if binary:
        labels_index.update({
            'O': np.nan,  # other
            '~': np.nan  # noise
        })
    else:
        raise NotImplementedError('Multiclass classification not implemented.')
    labels_file = data_dir_path / labels_csv_file
    labels_df = pd.read_csv(labels_file, header=None)
    labels = labels_df.iloc[:, 1]
    labels.index = labels_df.iloc[:, 0]
    features_glob_pattern = str(data_dir_path / features_dir / '*.mat')
    files = sorted(glob.glob(features_glob_pattern))
    if num_files_to_read is not None:
        files= files[:num_files_to_read]
    pids = []
    X_in = []
    y_in = []
    for f in files:
        pid = pathlib.Path(f).stem
        label = labels[pid]
        int_label = labels_index[label]
        if pd.isna(int_label):
            continue
        pids.append(pid)
        y_in.append(int_label)
        x_in = scipy.io.loadmat(f)['val'][0]
        X_in.append(x_in)
    pids_chunked = []
    X_chunked = []
    y_chunked = []
    for xx, yy, pp in zip(X_in, y_in, pids):
        xxx = xx.copy()
        new_chunks = []
        while True:
            if len(xxx) < slice_length:
                break
            new_chunks.append(xxx[:slice_length])
            if len(xxx) == slice_length:
                break
            xxx = xxx[slice_length:]
        X_chunked += new_chunks
        y_chunked.append(np.repeat(yy, len(new_chunks)))
        pids_chunked.append(np.repeat(pp, len(new_chunks)))
    X_chunked = np.array(X_chunked)
    y_chunked = np.concatenate(y_chunked)
    return X_chunked, y_chunked
