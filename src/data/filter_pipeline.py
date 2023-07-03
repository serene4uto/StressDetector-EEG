import mne


def fp1(epoch_data):
    "Apply 4th-order Butterworth Bandpass filter 1-45 Hz"
    epoch_data.filter(1, 45, method='iir', iir_params=dict(ftype='butter', order=4))



FILTER_PIPELINE = {
    'fp1' : fp1
}
    
