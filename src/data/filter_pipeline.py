import mne


def fp1(epoch_data):
    '''
        1. Apply 4th-order Butterworth Bandpass filter 1-45 Hz
    '''
    epoch_data.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4))


def fp2(epoch_data):
    '''
        1. Apply 4th-order Butterworth Bandpass filter 1-45 Hz.
        2. Apply Savitzky-Golay with h_freq = 1.
    '''
    epoch_data.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4))
    
    epoch_data.savgol_filter(h_freq=1, verbose=True)



FILTER_PIPELINE = {
    'NoFilter': None,
    'fp1' : fp1,
    'fp2' : fp2,
}
    
