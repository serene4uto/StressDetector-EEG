import mne
from scipy.stats import kurtosis

def ica_filter(epoch_data, kurtosis_threshold = 1):
    ica = mne.preprocessing.ICA(
        n_components=15, method="fastica",
        # max_iter=10000,
        max_iter="auto", 
        random_state=97
    )
    ica.fit(epoch_data, verbose=False)

    # Detect ICs containing blink artifact by using kurtosis
    blink_ic_idx = []
    ic_sources = ica.get_sources(epoch_data)
    for k_idx, k_value in enumerate(kurtosis(ic_sources.get_data().T)):
        if k_value > kurtosis_threshold:
            blink_ic_idx.append(k_idx)

    # Detect ICs containing muscle artifact automatically
    muscle_idx_auto, scores = ica.find_bads_muscle(epoch_data)

    artifact_list = blink_ic_idx + [x for x in muscle_idx_auto if x not in blink_ic_idx]

    # Apply on original data to filter noise:
    return ica.apply(epoch_data, exclude=artifact_list, verbose=False)





def pp1(epoch_set):
    '''
        1. Apply 4th-order Butterworth Bandpass filter 1-45 Hz
    '''
    epoch_set.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4))

    return epoch_set


def pp2(epoch_set):
    '''
        1. Apply 4th-order Butterworth Bandpass filter 1-45 Hz.
        2. Apply Savitzky-Golay with h_freq = 1.
    '''
    epoch_set.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4))
    
    epoch_set.savgol_filter(h_freq=1, verbose=True)

    return epoch_set

def pp3(epoch_set):
    '''
        1. Apply 4th-order Butterworth Bandpass filter 1-45 Hz.
        2. Apply ICA (fastica) to detect artifact components (ocular, muscle).
        3. Reconstruct EEG signal without detected artifcat components.
    '''

    epoch_set.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4), verbose=False)

    new_epoch_set = []
    for epoch_idx in range(len(epoch_set)):
        new_epoch_set.append(ica_filter(epoch_set[epoch_idx], 1))
    # print(new_epoch_set)

    return mne.concatenate_epochs(new_epoch_set)



PREPROCESSING_PIPELINES = {
    'pp1' : pp1,
    'pp2' : pp2,
    'pp3' : pp3,
}
    
