import os
import numpy as np
import pandas as pd
import mne
import scipy

DATA_DIR_ORIGIN = "data/origin"
DATA_DIR_RAW = DATA_DIR_ORIGIN + "/raw_data"
LABEL_PATH = DATA_DIR_ORIGIN + "/scales.xls"
COLUMNS_TO_RENAME = {
    'Subject No.': 'subject_no',
    'Trial_1'   : 't1_Arithmetic',
    'Unnamed: 2': 't1_Mirror',
    'Unnamed: 3': 't1_Stroop',
    'Trial_2'   : 't2_Arithmetic',
    'Unnamed: 5': 't2_Mirror',
    'Unnamed: 6': 't2_Stroop',
    'Trial_3'   : 't3_Arithmetic',
    'Unnamed: 8': 't3_Mirror',
    'Unnamed: 9': 't3_Stroop'
}

EXP_TYPES = ["Arithmetic", "Mirror", "Stroop"]

class Dataset:
    # Parameters

    def __init__(self, exp_type, subject_list=np.arange(1,41), 
                 trial_list=np.arange(1,4), channel_list=None,
                 epoch_sec=0, overlap_rate=0.5):

        self.fs = 128
        self.channel_type = 'eeg'

        if channel_list == None:
            self.channel_list = self.get_full_channels()
            self.n_channels = 32
        else :
            self.channel_list = channel_list
            self.n_channels = len(self.channel_list)
        
        self.filter_low_freq = 1.0
        self.filter_high_freq = 50.0
        self.filter_order = 4


        self.exp_type = exp_type
        self.subject_list = subject_list
        self.trial_list = trial_list
        self.epoch_sec = epoch_sec
        self.overlap_rate = overlap_rate

        self.label_annotation = {
            "Non-Stress" : 0,
            "Stress"     : 1
        }

        self.data = None

        # Set mne info
        self.info = mne.create_info(ch_names = self.channel_list, sfreq = self.fs, ch_types = self.channel_type)
        # Set montage
        self.montage = mne.channels.make_standard_montage('standard_1020')

        self.make_dataset() # Load and processing data

        pass

    def make_dataset(self):
        dir = DATA_DIR_RAW
        data_key = 'Data'

        chosen_labels = self.load_labels()
        # dataset_temp = [[] for _ in range(len(self.subject_list))]

        self.data = []

        for filename in os.listdir(dir):
            for subject_idx, subject in enumerate(self.subject_list):
                trial_set = []
                for trial_idx, trial in enumerate(self.trial_list):
                    if f"{self.exp_type}_sub_{subject}_trial{trial}" in filename:
                        f = os.path.join(dir, filename)

                        chosen_channel_idx = np.where(np.isin(self.get_full_channels(), self.channel_list))[0].tolist()

                        # Load & Convert to mne RawArray
                        loaded_data = mne.io.RawArray(
                            scipy.io.loadmat(f)[data_key][chosen_channel_idx], self.info, first_samp=0, copy='auto', verbose=None
                        )
                        # if self.epoch_sec == 0:
                        #     self.data[self.subject_list.tolist().index(subject)].append(
                                
                        #         loaded_data.filter(self.filter_low_freq, self.filter_high_freq, 
                        #                      method='iir', iir_params=dict(ftype='butter', order=self.filter_order))
                        #     )
                        # else:
                        trial_set.append(
                            # Segment to epochs
                            # Apply band-pass filter (1-50 Hz) on each epoch
                            mne.make_fixed_length_epochs(
                                raw=loaded_data, duration=self.epoch_sec, 
                                preload=True, overlap=self.overlap_rate,
                                id = int(chosen_labels.loc[subject, f"t{trial}_{self.exp_type}"]),
                                ).filter(self.filter_low_freq, self.filter_high_freq, 
                                         method='iir', iir_params=dict(ftype='butter', order=self.filter_order))
                        )

                if trial_set:
                    temp_set = mne.concatenate_epochs(trial_set).set_montage(self.montage)
                    temp_set.event_id = self.label_annotation
                    self.data.append(temp_set)
        pass
    
    def load_labels(self):

        labels = pd.read_excel(LABEL_PATH)
        labels = labels.rename(columns=COLUMNS_TO_RENAME)
        labels = labels[1:]
        labels = labels.astype("int")
        labels = labels > 5
        
        chosen_key = []

        
        for trial in self.trial_list:
            chosen_key.append(f"t{trial}_{self.exp_type}")
    
        return labels[chosen_key].loc[self.subject_list]



    def get_full_channels(self):
        coordinates_file = os.path.join(DATA_DIR_ORIGIN, "Coordinates.locs") 

        channel_names = []

        with open(coordinates_file, "r") as file:
            for line in file:
                elements = line.split()
                channel = elements[-1]
                channel_names.append(channel)
                
        return channel_names




     