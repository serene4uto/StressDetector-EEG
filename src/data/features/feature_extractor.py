import mne
import numpy as np
from mne_features.feature_extraction import extract_features

"""
    Description for functions excuting feature extraction on an epoch:
    1. Extract feature on each channel of epoch
    2. Transpose to 2D shape [channels x features]
"""


"""--------------------Time-Series---------------------------"""
def epoch_feature_raw(epoch, metadata):
    return epoch[None, :].squeeze()

def epoch_feature_mean(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'mean'}).transpose()

def epoch_feature_std(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'std'}).transpose()

def epoch_feature_rms(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'rms'}).transpose()

def epoch_feature_ptp(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'ptp_amp'}).transpose()

def epoch_feature_kurtosis(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'kurtosis'}).transpose()

def epoch_feature_hjorth_complexity(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'hjorth_complexity'}).transpose()



"""--------------------Entropy---------------------------"""

def epoch_feature_spect_entropy(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'spect_entropy'}).transpose()

def epoch_feature_svd_entropy(epoch, metadata):
    return extract_features(epoch[None, :], metadata['sfreq'], {'svd_entropy'}).transpose()


"""--------------------Time-Frequency---------------------------"""

def epoch_feature_eegband_psd_welch(epoch, metadata):
    # _, n_channels, n_tpoint = epoch.get_data().shape
    sfreq = metadata['sfreq']
    n_per_seg = int(sfreq)
    n_fft = int(n_per_seg*2)
    psds, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq = sfreq, output = 'power',
                             n_fft = n_fft, n_overlap = int(0.5*n_per_seg), n_per_seg = n_per_seg)

    max_f = freqs[-1]

    freq_bands = {
        'Delta' : (0.5  , 4), 
        'Theta' : (4    , 8), 
        'Alpha' : (8    , 13), 
        'Beta'  : (13   , 30), 
        'Gamma' : (30   , 100 if max_f > 100 else max_f)}
    
    band_psd = []

    # Compute the band psd for each frequency band within each epoch
    for band, (fmin, fmax) in freq_bands.items():
        band_indices = np.where(np.logical_and(freqs >= fmin, freqs <= fmax))[0]
        band_psd.append(np.sum(psds[:, band_indices], axis=1).squeeze())
    
    return np.array(band_psd).T


def epoch_feature_eegband_psd_multitaper(epoch, metadata):
    # _, n_channels, n_tpoint = epoch.get_data().shape
    sfreq = metadata['sfreq']
    psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, sfreq = sfreq, output = 'power')

    max_f = freqs[-1]

    freq_bands = {
        'Delta' : (0.5  , 4), 
        'Theta' : (4    , 8), 
        'Alpha' : (8    , 13), 
        'Beta'  : (13   , 30), 
        'Gamma' : (30   , 100 if max_f > 100 else max_f)}
    
    band_psd = []

    # Compute the band psd for each frequency band within each epoch
    for band, (fmin, fmax) in freq_bands.items():
        band_indices = np.where(np.logical_and(freqs >= fmin, freqs <= fmax))[0]
        band_psd.append(np.sum(psds[:, band_indices], axis=1).squeeze())
    
    return np.array(band_psd).T


def epoch_feature_rbpsd_tab_welch(epoch, metadata):
    '''
        Calculate psd of three band theta (4-8 Hz), alpha (8-12 Hz), beta (13-30 Hz) using welch method
    '''
    
    tab_psd = epoch_feature_eegband_psd_welch(epoch, metadata)[:, [2,3,4]]
    total_power = np.sum(tab_psd, axis = 1).reshape(-1, 1)

    rbptab_psd = tab_psd / total_power

    return rbptab_psd


def epoch_feature_rbpsd_tab_multitaper(epoch, metadata):
    '''
        Calculate psd of three band theta (4-8 Hz), alpha (8-12 Hz), beta (13-30 Hz) using multitaper method
    '''

    tab_psd = epoch_feature_eegband_psd_multitaper(epoch, metadata)[:, [2,3,4]]
    total_power = np.sum(tab_psd, axis = 1).reshape(-1, 1)

    rbptab_psd = tab_psd / total_power

    return rbptab_psd


SCALAR_FEATURE_EXTRACTOR = {
    'raw'               : epoch_feature_raw,

    # Time-Series
    'mean'              : epoch_feature_mean,
    'std'               : epoch_feature_std,
    'rms'               : epoch_feature_rms,
    'peak_to_peak'      : epoch_feature_ptp,
    'kurtosis'          : epoch_feature_kurtosis,
    'hjorth_complexity' : epoch_feature_hjorth_complexity,

    # Entropy
    'spect_entropy'      : epoch_feature_spect_entropy,
    'svd_entropy'        : epoch_feature_svd_entropy,

    # Time-Frequency
    'psd_eegband_welch'         : epoch_feature_eegband_psd_welch,
    'psd_eegband_multitaper'    : epoch_feature_eegband_psd_multitaper,
    
    'rbpsd_tab_welch'           : epoch_feature_rbpsd_tab_welch, 
    'rbpsd_tab_multitaper'      : epoch_feature_rbpsd_tab_multitaper, 
}

IMAGE_FEATURE_EXTRACTOR = {

}

def extract_scalar_feature_set(features, epoch_set, metadata):
    '''
        Extracting Feature Set.
        Args:
            features : (list) desired features to extract.

            epoch_set : subject (list) --> EEG data (mne.Epochs).

            metadata : (dict) metadata of epoch_set.
        
        Output:
            feature_set: subject (list)--> classes (list) --> epochs (list) -->computed feature set (ndarray: channel x features).
    '''
    feature_set = []
    for subj_idx, subj_id in enumerate(metadata['subjects']):
        class_set = [[] for _ in metadata['classes']]
        for epoch_idx, epoch in enumerate(epoch_set[subj_idx]):
            ftr_set = []
            for ftr in features:
                ftr_set.append(SCALAR_FEATURE_EXTRACTOR.get(ftr)(epoch, metadata))
            
            class_set[epoch_set[subj_idx].events[epoch_idx, -1]-1].append(
                np.concatenate(ftr_set, axis=1)
            )

        feature_set.append(class_set)
    return feature_set







# def extract_feature_set(feature, epoch_data, metadata):
#     feature_set = []
#     for subj_idx, subj_id in enumerate(metadata['subjects']):
#         class_set = [[] for _ in metadata['classes']]
#         for epoch_idx, epoch in enumerate(epoch_data[subj_idx]):

#             class_set[epoch_data[subj_idx].events[epoch_idx, -1]-1].append(
#                 FEATURE_EXTRACTOR.get(feature)(epoch, epoch_data[subj_idx].info['sfreq'])
#             )

#         feature_set.append(class_set)
#     return feature_set

# def extract_feature_Mean(epoch_data, metadata):
#     feature_set = []
#     for subj_idx, subj_id in enumerate(metadata['subjects']):
#         class_set = [[] for _ in metadata['classes']]
#         for epoch_idx, epoch in enumerate(epoch_data[subj_idx]):
#             # calculate Mean
#             class_set[epoch_data[subj_idx].events[epoch_idx, -1]-1].append(
#                 extract_features(epoch[None, :], epoch_data[subj_idx].info['sfreq'], {'mean'})
#             )
#         feature_set.append(class_set)
#     return feature_set

# def extract_feature_RMS(epoch_data, metadata):
#     feature_set = []
#     for subj_idx, subj_id in enumerate(metadata['subjects']):
#         class_set = [[] for _ in metadata['classes']]
#         for epoch_idx, epoch in enumerate(epoch_data[subj_idx]):
#             # calculate RMS
#             class_set[epoch_data[subj_idx].events[epoch_idx, -1]-1].append(
#                 extract_features(epoch[None, :], epoch_data[subj_idx].info['sfreq'], {'rms'})
#             )
#         feature_set.append(class_set)
#     return feature_set

# def extract_feature_PTP(epoch_data, metadata):
#     feature_set = []
#     for subj_idx, subj_id in enumerate(metadata['subjects']):
#         class_set = [[] for _ in metadata['classes']]
#         for epoch_idx, epoch in enumerate(epoch_data[subj_idx]):
#             # calculate Peak-to-Peak
#             class_set[epoch_data[subj_idx].events[epoch_idx, -1]-1].append(
#                 extract_features(epoch[None, :], epoch_data[subj_idx].info['sfreq'], {'ptp_amp'})
#             )
#         feature_set.append(class_set)
#     return feature_set