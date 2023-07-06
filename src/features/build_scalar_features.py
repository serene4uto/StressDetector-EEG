import os
import argparse
import pickle
import mne
from pathlib import Path
import shutil

import feature_extractor as fe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str, default=None, 
            help="The path for input dataset")
    
    parser.add_argument('--features', nargs='+', type=str, default=None, choices=fe.FEATURE_EXTRACTOR.keys(), 
            help="Features to extract")
    
    parser.add_argument('--save-path', type=str, default=None, 
            help="The path for saving")
    
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
            help="Overwrite cache or not")

    args = parser.parse_args()

    dataset_dir_path = args.dataset_path
    
    # read metadata
    loaded_metadata = None
    with open(dataset_dir_path + "/metadata.pkl", 'rb') as file:
        loaded_metadata = pickle.load(file)

    loaded_dataset = []
    for subject_idx, subject_id in enumerate(loaded_metadata['subjects']):
        loaded_dataset.append(mne.read_epochs(dataset_dir_path + f'/sub{subject_id}_epo.fif', verbose = False))


    if args.save_path:
        save_dir_path = args.save_path + '/features/scalar'
    else:
        save_dir_path = dataset_dir_path + '/features/scalar'
    
    features_to_extract = args.features

    if os.path.exists(save_dir_path):
        if args.overwrite:
            shutil.rmtree(save_dir_path) 
            Path(save_dir_path).mkdir(parents=True, exist_ok=False)
        else:
            # check cached
            for filename in os.listdir(save_dir_path):
                print(filename)
                ft_name = filename.replace(".pkl", "")
                if ft_name in features_to_extract:
                    features_to_extract.remove(ft_name)
    else:
        Path(save_dir_path).mkdir(parents=True, exist_ok=False)
    


    for ft in features_to_extract:
        feature_set = fe.extract_scalar_feature_set([ft], loaded_dataset, loaded_metadata)
        with open(f"{save_dir_path}/{ft}.pkl", "wb") as f:
            pickle.dump(feature_set, f)

    # Handle  metadata
    # available_scalar_f = []
    # for 
    # loaded_metadata['scalar_f'] = 
    # with open(dataset_dir_path + "/metadata.pkl", 'wb') as file:
    #     loaded_metadata = pickle.load(file)

    pass