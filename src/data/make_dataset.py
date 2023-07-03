import os
import numpy as np
import mne
import argparse
import logging
import pickle
import shutil

import dataloader
import vardefine as v

import filter_pipeline as fp

from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', type=str, default=v.DTS_TASK_LIST, choices=v.DTS_TASK_LIST, 
            help="Select tasks")
    parser.add_argument('--subjects', nargs='+', type=int, default=np.arange(1, v.DTS_MAX_SUBJECT+1).tolist(),
                        choices=np.arange(1, v.DTS_MAX_SUBJECT+1).tolist(), help="Select subjects")
    # parser.add_argument('--trials', nargs='+', type=int, default=v.DTS_TRIAL_LIST,
    #                     choices=v.DTS_TRIAL_LIST, help="Select trials")
    parser.add_argument('--channels', nargs='+', type=str, default=v.DTS_CHANNEL_LIST,
                        choices=v.DTS_CHANNEL_LIST, help="Select channels")
    
    parser.add_argument('--filter-pipeline', type=str, default=None, 
                        choices=fp.FILTER_PIPELINE.keys(), help="Choose a filter pipeline")
    
    parser.add_argument('--epoch-duration', type=float, default=v.DTS_MAX_TASK_DURATION_SEC, help="Set epoch duration in seconds to segment")
    parser.add_argument('--overlap-rate', type=float, default=0, help="Set epoch overlap rate to segment")

    parser.add_argument('--save-path', type=str, default=v.DTS_DIR_PROCESSED, help="Indicate the path to save")
    
    args = parser.parse_args()

    print(f"Selected sublist: {args}")

    # Create Meta
    metadata = {
        "tasks"     : v.DTS_TASK_LIST,
        "subjects"  : args.subjects,
        "trials"    : v.DTS_TRIAL_LIST,
        "channels"  : args.channels,
        "filters"   : args.filter_pipeline,
    } 

    # Check cache
    save_dir_path = f"{args.save_path}/{args.epoch_duration}sec_overlap{args.overlap_rate}_{args.filter_pipeline}"
    save_metadata_file_path = save_dir_path + f"/metadata.pkl"
    cached = False
    
    if os.path.exists(save_dir_path):
        with open(save_metadata_file_path, 'rb') as file:
            loaded_metadata = pickle.load(file)
            if loaded_metadata == metadata:
                cached = True

        if cached == False:
            shutil.rmtree(save_dir_path) 
            

    if cached == False:
        Path(save_dir_path).mkdir(parents=True, exist_ok=False)
        # Save Metadata
        with open(save_metadata_file_path, 'wb') as f:
            pickle.dump(metadata, f)

        # load dataset
        dataset = dataloader.load_data( task_list = args.tasks, 
                                        subject_list=args.subjects,
                                        trial_list=v.DTS_TRIAL_LIST,
                                        channel_list=args.channels)
        
        # load origin labels
        labels_df = dataloader.load_labels(task_list = args.tasks, 
                                            subject_list=args.subjects,
                                            trial_list=v.DTS_TRIAL_LIST,
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
                for trial_idx, trial in enumerate(v.DTS_TRIAL_LIST):
                    # Convert to mne RawArray
                    temp = mne.io.RawArray( data = dataset[subject_idx][task_idx][trial_idx], 
                                    info=info, first_samp=0, copy='auto', verbose=None)

                    epoch_temp = mne.make_fixed_length_epochs(
                            raw=temp, duration=args.epoch_duration,
                            preload=True, overlap=args.overlap_rate,
                            id = int(labels_df.loc[subject_id, f"t{trial}_{task}"]),
                        )

                    # Filter
                    if args.filter_pipeline:
                        fp.FILTER_PIPELINE.get(args.filter_pipeline)(epoch_temp)

                    trial_set.append(epoch_temp)
      
                temp_set = mne.concatenate_epochs(trial_set).set_montage(montage) 
                temp_set.event_id = event_id
                task_set.append(temp_set)    

            epoch_dataset.append(mne.concatenate_epochs(task_set))    


        # save data
        for subject_idx, subject_id in enumerate(args.subjects): 
            epoch_dataset[subject_idx].save(save_dir_path + f"/sub{subject_id}_epo.fif", overwrite=True)

    pass