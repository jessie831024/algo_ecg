import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d
from scipy.stats import zscore
from scipy.integrate import trapz

SAMPLING_RATE = 300 

def peak_detect_return_rr(ecg): 
    # linear spaced vector between 0.5 pi and 1.5 pi 
    v = np.linspace(0.5 * np.pi, 1.5 * np.pi, 15)

    # create sine filter for approximating QRS feature
    peak_filter = np.sin(v)

    ecg_transformed = np.correlate(ecg, peak_filter, mode="same")
    rr_peaks, _ = find_peaks(ecg_transformed, distance=SAMPLING_RATE*(30/60))
    rr_ecg = np.diff(rr_peaks)
    return rr_ecg

def timedomain(rr):
    results = {}
    rr_in_ms = (rr / SAMPLING_RATE) * 1000 
    hr = 60 * SAMPLING_RATE /rr
    results['Mean_RR_ms'] = np.mean(rr_in_ms)
    results['STD_RR_ms'] = np.std(rr_in_ms)
    results['Mean_HR_beats_per_min_kubio'] = 60 * SAMPLING_RATE/np.mean(rr)
    results['Mean_HR_beats_per_min'] = np.mean(hr)
    results['STD_HR_beats_per_min'] = np.std(hr)
    results['Min_HR_beats_per_min'] = np.min(hr) 
    results['Max_HR_beats_per_min'] = np.max(hr)
    results['RMSSD_ms'] = np.sqrt(np.mean(np.square(np.diff(rr_in_ms))))
    results['NN50'] = np.sum(np.abs(np.diff(rr_in_ms)) > 50)*1
    results['pNN50'] = 100 * np.sum((np.abs(np.diff(rr_in_ms)) > 50)*1) / len(rr_in_ms)
    results['NN70'] = np.sum(np.abs(np.diff(rr_in_ms)) > 70)*1
    results['pNN70'] = 100 * np.sum((np.abs(np.diff(rr_in_ms)) > 70)*1) / len(rr_in_ms)
    return results


def freq_domain(fxx, pxx):

    #frequency bands: very low frequency (VLF), low frequency (LF), high frequency (HF) 
    cond_VLF = (fxx >= 0) & (fxx < 0.04)
    cond_LF = (fxx >= 0.04) & (fxx < 0.15)
    cond_HF = (fxx >= 0.15) & (fxx < 0.4)

    #calculate power in each band by integrating the spectral density using trapezoidal rule 
    VLF = trapz(pxx[cond_VLF], fxx[cond_VLF])
    LF = trapz(pxx[cond_LF], fxx[cond_LF])
    HF = trapz(pxx[cond_HF], fxx[cond_HF])

    #total power sum
    total_power = VLF + LF + HF

    # calculate power in each band by integrating the spectral density 
    vlf = trapz(pxx[cond_VLF], fxx[cond_VLF])
    lf = trapz(pxx[cond_LF], fxx[cond_LF])
    hf = trapz(pxx[cond_HF], fxx[cond_HF])


    #peaks (Hz) in each band
    peak_VLF = fxx[cond_VLF][np.argmax(pxx[cond_VLF])]
    peak_LF = fxx[cond_LF][np.argmax(pxx[cond_LF])]
    peak_HF = fxx[cond_HF][np.argmax(pxx[cond_HF])]

    #fractions
    LF_nu = 100 * lf / (lf + hf)
    HF_nu = 100 * hf / (lf + hf)

    results = {}
    results['Power_VLF_ms2'] = VLF
    results['Power_LF_ms2'] = LF
    results['Power_HF_ms2'] = HF   
    results['Power_Total_ms2'] = total_power

    results['LF_HF_ratio'] = (LF/HF)
    results['Peak_VLF_Hz'] = peak_VLF
    results['Peak_LF_Hz'] = peak_LF
    results['Peak_HF_Hz'] = peak_HF

    results['Fraction_LF_nu'] = LF_nu
    results['Fraction_HF_nu'] = HF_nu

    return results


def calculate_hrv_based_on_peak_intervals3 (row): 
    rr_ecg = peak_detect_return_rr(row)
    time_domain_features = timedomain(rr_ecg)
    for k, v in time_domain_features.items(): 
        row[k] = v
    return row[time_domain_features.keys()]

def filter_interpolation_calculate_fre_domain_features (row): 
    ## rr_ecg == nni from the peak_detect_return_rr return 
    rr_ecg = peak_detect_return_rr(row)
    # fit function to the dataset
    x_ecg = np.cumsum(rr_ecg)/300 
    f_ecg = interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')
    
    # sample rate for interpolation
    fs = 4
    steps = 1 / fs
    
    # sample using the interpolation function
    xx_ecg = np.arange(0, np.max(x_ecg), steps)
    rr_interpolated_ecg = f_ecg(xx_ecg)

    rr_ecg[np.abs(zscore(rr_ecg)) > 2] = np.median(rr_ecg)
    x_ecg = np.cumsum(rr_ecg)/300
    f_ecg = interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')

    xx_ecg = np.arange(0, np.max(x_ecg), steps)
    clean_rr_interpolated_ecg = f_ecg(xx_ecg)

    fxx, pxx = welch(x=clean_rr_interpolated_ecg, fs=fs, nperseg=len(clean_rr_interpolated_ecg))
    powerspectrum_f = interp1d(fxx, pxx, kind='cubic', fill_value= 'extrapolate')
    fre_domain_features = freq_domain(fxx, pxx)
    
    for k, v in fre_domain_features.items(): 
        row[k] = v
    return row[fre_domain_features.keys()]
    
def generate_time_fre_domain_features(row): 
#    time_domain = calculate_hrv_based_on_peak_intervals(row)
    time_domain = calculate_hrv_based_on_peak_intervals3(row)
    fre_domain = filter_interpolation_calculate_fre_domain_features(row)
    result = pd.concat([time_domain, fre_domain], axis=0)
    return result 
    
def functional_return_all_features(X, axis = 1): 
    features = X.apply(generate_time_fre_domain_features, axis) 
    return features
    
def calculate_hrv_based_on_peak_intervals (row): 
    peaks, _ = find_peaks(row, height=0, distance=150)
    nni = np.diff(peaks)
    nnid = np.diff(nni)    
#    nnid_squared = np.square(nnid)
    avg_nni = np.mean(nni)
    std_nni = np.std(nni)
    rmssd = np.sqrt(np.mean(np.square(nnid)))
    rmssd_nn = rmssd / avg_nni
    pnn70 = len(nnid[np.abs(nnid) > 70])
    row['avg_nni'] = avg_nni
    row['std_nni'] = std_nni
    row['rmssd'] = rmssd
    row['rmssd_nn'] = rmssd_nn
    row['pnn70'] = pnn70
    return row[['avg_nni',	'std_nni',	'rmssd',	'rmssd_nn',	'pnn70']]

def select_cols (X, col_names): 
    return X[col_names]


class AllFeatureCustomTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis=1):
        self.axis = axis
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.apply(generate_time_fre_domain_features, self.axis) 
        return X

class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.correlation_matrix = None
        self.features_to_drop = []

    def fit(self, X, y=None):
        # Compute the correlation matrix
        self.correlation_matrix = X.corr()

        # Identify features with high correlation
        upper_triangle = np.triu(np.ones(self.correlation_matrix.shape), k=1)
        correlated_features = np.where(np.abs(self.correlation_matrix) > self.threshold * upper_triangle)

        # Create a set of unique features to drop
        self.features_to_drop = set()
        for feature1, feature2 in zip(*correlated_features):
            if feature1 != feature2:
                self.features_to_drop.add(feature2)

        return self

    def transform(self, X, y=None):
        # Drop the highly correlated features
        X_transformed = X.drop(columns=list(self.features_to_drop), errors='ignore')
        return X_transformed
