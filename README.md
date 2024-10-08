# algo-ecg
Algorithms for afib classification based on ecg data

## Data 

AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017
https://physionet.org/content/challenge-2017/1.0.0/

The dataset used for this teaching module is based on the publicly available CinC 2017 via PhysioNet. 
Please download and create a symlink to the ./data folder. 
Please note that entire training2017 directory contains 8528 data files totalling 2.1GB. 
The import script allows you to select a smaller number (1000) of files for teaching purpose. 

The study contains different types of cardiac arrhythmias, but for the purpose of the teaching module we are only interested in atrial fibrillation. 
The training set contains 8,528 single lead ECG recordings (collected using AliveCor) lasting from 9 s to just over 60 s. 
ECG recordings were sampled as 300 Hz and they have been band pass filtered by the AliveCor device.

## Setup 

set up a virtual environment 
```
pyenv virtualenv 3.8.10 cinc
```
follow this if you don't have pyenv on your machine: https://github.com/pyenv/pyenv-virtualenv

install all dependencies 
```
pip install -r requirements.txt 
```

install this package

```
pip install -e . 
```

## Teaching schedule 

### [Week 1](https://github.com/jessie831024/algo_ecg/blob/main/reading/w1.md): 
- Data exploration 
- Feature engineering 
- Initial pipeline setup 

### [Week 2](https://github.com/jessie831024/algo_ecg/blob/main/reading/w2.md): 
- Logistic regression gradient descent
- Support vector machine gradient descent
- Decision tree & random forest

### [Week 3](https://github.com/jessie831024/algo_ecg/blob/main/reading/w3.md): 
- Hyperparameter tuning & Classifier comparison
- Deep learning 
  - Feedforward 
  - Convolutional neural network 
  - Recurrent neural network
- Detecting AFib using photoplethysmography data ([paper](https://arxiv.org/abs/1807.10707))



