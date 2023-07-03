import os
import numpy as np
import mne
import argparse
import logging

import dataloader
import vardefine as v

if __name__ == "__main__":

    # logger = logging.getLogger()
    # # Configure a handler to print logs to the console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', type=str, default=v.DTS_TASK_LIST, choices=v.DTS_TASK_LIST, 
            help="Select tasks")
    parser.add_argument('--subjects', nargs='+', type=int, default=np.arange(1, v.DTS_MAX_SUBJECT+1).tolist(),
                        choices=np.arange(1, v.DTS_MAX_SUBJECT+1).tolist(), help="Select subjects")
    parser.add_argument('--trials', nargs='+', type=int, default=np.arange(1, v.DTS_MAX_TRIAL+1).tolist(),
                        choices=np.arange(1, v.DTS_MAX_TRIAL+1).tolist(), help="Select trials")
    parser.add_argument('--channels', nargs='+', type=str, default=v.DTS_CHANNEL_LIST,
                        choices=v.DTS_CHANNEL_LIST, help="Select channels")
    
    # parser.add_argument('--filter', type=bool, default=True, help="Enable Butterworth Bandpass filter (1-50Hz)") #TODO: make it become processing pipeline

    parser.add_argument('--epoch-duration', type=float, default=v.DTS_MAX_TASK_DURATION_SEC, help="Set epoch duration in seconds to segment")
    parser.add_argument('--overlap-rate', type=float, default=0, help="Set epoch overlap rate to segment")

    parser.add_argument('--save-path', type=str, default=v.DTS_DIR_PROCESSED, help="Indicate the path to save")
    
    args = parser.parse_args()

    print(f"Selected sublist: {args}")

    # save_path = args.save_path + "/"
    # if args.filter:
    #     save_path += "filtered"


    # Create Meta
    metadata = {
        "tasks"     : args.tasks,
        "subjects"  : args.subjects,
        "trials"    : args.trials,
        "channels"  : args.channels,
    }

    print(metadata)


    # load dataset
    dataset = dataloader.load_data( task_list = args.tasks, 
                                    subject_list=args.subjects,
                                    trial_list=args.trials,
                                    channel_list=args.channels)
    
    # load origin labels
    labels_df = dataloader.load_labels(task_list = args.tasks, 
                                        subject_list=args.subjects,
                                        trial_list=args.trials,
                                        stress_lvl_threshold=5)
    
    print(dataset)
    
    # Convert to MNE epoch dataset
    info = mne.create_info(ch_names = args.channels, sfreq = v.DTS_SAMPLING_FREQ, ch_types = 'eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    
    epoch_dataset = []
    event_id = {}
    for cl in v.DTS_CLASS_LIST:
        event_id[cl] = v.DTS_CLASS_LIST.index(cl)

    for subject_idx, subject_id in enumerate(args.subjects):
        task_set = []
        for task_idx, task in enumerate(args.tasks):
            trial_set = []
            for trial_idx, trial in enumerate(args.trials):
                # Convert to mne RawArray
                temp = mne.io.RawArray( data = dataset[subject_idx][task_idx][trial_idx], 
                                info=info, first_samp=0, copy='auto', verbose=None)
                
                trial_set.append(mne.make_fixed_length_epochs(
                        raw=temp, duration=args.epoch_duration,
                        preload=True, overlap=args.overlap_rate,
                        id = int(labels_df.loc[subject_id, f"t{trial}_{task}"]),
                    )
                )
                

            temp_set = mne.concatenate_epochs(trial_set).set_montage(montage) 
            temp_set.event_id = event_id
            task_set.append(temp_set)    

        epoch_dataset.append(task_set)     

    print(epoch_dataset)
    pass