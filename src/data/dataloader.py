import os
import numpy as np
import pandas as pd
import mne
import scipy
import vardefine as v

def get_full_channels(self):
    coordinates_file = os.path.join(v.DTS_DIR_ORIGIN, "Coordinates.locs") 
    channel_names = []
    with open(coordinates_file, "r") as file:
        for line in file:
            elements = line.split()
            channel = elements[-1]
            channel_names.append(channel)
            
    return channel_names

def load_data(task_list, subject_list, trial_list, channel_list):
    dir = v.DTS_DIR_ORIGIN_RAW
    data_key = 'Data'

    # Load
    dataset = []
    channel_idx_list = np.where(np.isin(v.DTS_CHANNEL_LIST, channel_list))[0].tolist()
    for subject_idx, subject_id in enumerate(subject_list):
        task_set = [[] for _ in task_list]            
        for task_idx, task in enumerate(task_list):
            for trial in trial_list:
                filename = f"{task}_sub_{subject_id}_trial{trial}.mat"
                f = os.path.join(dir, filename)
                task_set[task_idx].append(scipy.io.loadmat(f)[data_key][channel_idx_list])

        dataset.append(task_set)

    return dataset

def load_labels(task_list, subject_list, trial_list, stress_lvl_threshold=v.DTS_STRESS_LVL_THRESHOLD):
    labels = pd.read_excel(v.DTS_ORIGIN_LABEL_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > stress_lvl_threshold

    chosen_key = []
    
    for task in task_list:
        if task != "Relax":
            for trial in trial_list:
                chosen_key.append(f"t{trial}_{task}")

    labels_df = labels[chosen_key].loc[subject_list].astype(int)
    
    if "Relax" in task_list:
        col_indx = task_list.index("Relax")
        # Add value label for Relax
        for trial_idx, trial in enumerate(trial_list):
            labels_df.insert(col_indx * len(trial_list) + trial_idx,  f"t{trial}_Relax", 2)  
    
    return labels_df






