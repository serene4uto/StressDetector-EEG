import mne
import numpy as np

def time_series_features(data):
    '''
    Computes the features variance, RMS and peak-to-peak amplitude using the package mne_features.

    Args:
        data (mne epochs): EEG data.

    Returns:
        ndarray: Computed features.

    '''

    
    return 