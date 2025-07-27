import os
import requests
import shutil
import logging
import rarfile
import numpy as np
import scipy
import pandas as pd
import pickle
import mne
from sklearn.model_selection import train_test_split

from . import variables as v

from ..preprocess import preprocessing_pipeline as pp
from ..features import feature_extractor as fe

# Configure the logging system
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create a logger
logger = logging.getLogger('SAM40 Dataset')


def get_full_channels(self):
    coordinates_file = os.path.join(v.SAM40_DIR_ORIGIN_RAW, "Coordinates.locs") 
    channel_names = []
    with open(coordinates_file, "r") as file:
        for line in file:
            elements = line.split()
            channel = elements[-1]
            channel_names.append(channel)
            
    return channel_names

def load_data(task_list, subject_list, trial_list, channel_list):
    dir = os.path.join(v.SAM40_DIR_ORIGIN_RAW, "raw_data")
    data_key = 'Data'

    # Load
    dataset = []
    channel_idx_list = np.where(np.isin(v.SAM40_CHANNEL_LIST, channel_list))[0].tolist()
    for subject_idx, subject_id in enumerate(subject_list):
        task_set = [[] for _ in task_list]            
        for task_idx, task in enumerate(task_list):
            for trial in trial_list:
                filename = f"{task}_sub_{subject_id}_trial{trial}.mat"
                f = os.path.join(dir, filename)
                task_set[task_idx].append(scipy.io.loadmat(f)[data_key][channel_idx_list])

        dataset.append(task_set)

    return dataset

def load_labels(task_list, subject_list, trial_list, stress_lvl_threshold=v.SAM40_STRESS_LVL_THRESHOLD):
    labels = pd.read_excel(v.SAM40_ORIGIN_LABEL_PATH)
    labels = labels.rename(columns=v.SAM40_COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > stress_lvl_threshold

    chosen_key = []
    
    for task in task_list:
        if task != "Relax":
            for trial in trial_list:
                chosen_key.append(f"t{trial}_{task}")

    labels_df = labels[chosen_key].loc[subject_list].astype(int).add(2)
    
    if "Relax" in task_list:
        col_indx = task_list.index("Relax")
        
        # Add value label for Relax
        for trial_idx, trial in enumerate(trial_list):
            labels_df.insert(col_indx * len(trial_list) + trial_idx,  f"t{trial}_Relax", 1)  
    
    return labels_df

def make_dataset(
    tasks: list = v.SAM40_TASK_LIST,
    subjects: list = v.SAM40_SUBJECT_LIST,
    # trials=v.SAM40_TRIAL_LIST,
    channels: list = v.SAM40_CHANNEL_LIST,
    epoch_duration: float = v.SAM40_MAX_TASK_DURATION_SEC,
    overlap_rate: float = 0.5,
    pp_pipeline: str = None,

    features: list[str] = ['raw'],
    f_overwrite: bool =  False,

    split_type: str = 'combined',
    **kwargs,
    ):

    if not os.path.isdir(v.SAM40_DIR_ORIGIN_RAW):
        # Download SAM40 dataset
        os.makedirs(v.SAM40_DIR_ORIGIN_RAW, exist_ok=True)
        download_file_path = os.path.join(v.SAM40_DIR_ORIGIN_RAW, "SAM40.rar")

        response = requests.get(v.SAM40_URL, stream=True)
        if response.status_code == 200:
            with open(download_file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"SAM40 Dataset is downloaded successfully.")
        else:
            logger.info(f"Failed to download SAM40 Dataset. Status code: {response.status_code}")

        # Unrar SAM40 dataset
    
        with rarfile.RarFile(download_file_path, "r") as rf:
            rf.extractall(v.SAM40_DIR_ORIGIN_RAW)
        logger.info(f"SAM40 Dataset is extracted successfully.")

        
        os.remove(download_file_path)

        for item in os.listdir(os.path.join(v.SAM40_DIR_ORIGIN_RAW, "Data")):
            shutil.move(os.path.join(v.SAM40_DIR_ORIGIN_RAW, "Data", item), v.SAM40_DIR_ORIGIN_RAW)
        shutil.rmtree(os.path.join(v.SAM40_DIR_ORIGIN_RAW, "Data"))
        logger.info(f"SAM40 Dataset is moved successfully.")
    else:
        logger.info(f"SAM40 Dataset existed.")

    # Preprocess origin SAM40 dataset

    # Create pp Meta
    pp_metadata = {
        "sfreq"             : v.SAM40_SAMPLING_FREQ_HZ,
        "tasks"             : tasks,
        "subjects"          : subjects,
        "trials"            : v.SAM40_TRIAL_LIST,
        "channels"          : channels,
        "classes"           : v.SAM40_CLASS_LIST,
        "preprocessing"     : pp_pipeline,
    } 

    # Check pp cache
    pp_save_dir_path = f"{v.SAM40_DIR_PREPROCESSED}/{epoch_duration}sec_overlap{overlap_rate}_{pp_pipeline}"
    pp_save_metadata_file_path = pp_save_dir_path + f"/metadata.pkl"
    pp_cached = False

    if os.path.exists(pp_save_dir_path):
        with open(pp_save_metadata_file_path, 'rb') as file:
            loaded_metadata = pickle.load(file)
            if loaded_metadata == pp_metadata:
                pp_cached = True
            
        if pp_cached == False:
            shutil.rmtree(pp_save_dir_path)
    
    pp_epoch_dataset = []
    if pp_cached == False:
        os.makedirs(pp_save_dir_path, exist_ok=True)

        # Save pp Metadata
        with open(pp_save_metadata_file_path, 'wb') as f:
            pickle.dump(pp_metadata, f)
        
        # Load data
        pp_dataset = load_data(  task_list = tasks, 
                                subject_list=subjects,
                                trial_list=v.SAM40_TRIAL_LIST,
                                channel_list=channels)
        
        pp_labels_df = load_labels(  task_list = tasks,
                                    subject_list=subjects,
                                    trial_list=v.SAM40_TRIAL_LIST,
                                    stress_lvl_threshold=5)
        
        # Convert to MNE epoch dataset
        info = mne.create_info(ch_names = channels, sfreq = v.SAM40_SAMPLING_FREQ_HZ, ch_types = 'eeg')
        montage = mne.channels.make_standard_montage('standard_1020')

        event_id = {}
        for cl in v.SAM40_CLASS_LIST:
            event_id[cl] = v.SAM40_CLASS_LIST.index(cl)+1


        for subject_idx, subject_id in enumerate(subjects):
            task_set = []
            for task_idx, task in enumerate(tasks):
                trial_set = []
                for trial_idx, trial in enumerate(v.SAM40_TRIAL_LIST):
                    # Convert to mne RawArray
                    temp = mne.io.RawArray( data = pp_dataset[subject_idx][task_idx][trial_idx], 
                                    info=info, first_samp=0, copy='auto', verbose=None)

                    epoch_temp = mne.make_fixed_length_epochs(
                            raw=temp, duration=epoch_duration,
                            preload=True, overlap=overlap_rate,
                            id = int(pp_labels_df.loc[subject_id, f"t{trial}_{task}"]),
                        ).set_montage(montage)

                    # Preprocessing
                    if pp_pipeline:
                        epoch_temp = pp.PREPROCESSING_PIPELINES.get(pp_pipeline)(epoch_temp)

                    trial_set.append(epoch_temp)
      
                temp_set = mne.concatenate_epochs(trial_set) 
                temp_set.event_id = event_id
                task_set.append(temp_set)    

            pp_epoch_dataset.append(mne.concatenate_epochs(task_set))
        
        # Save preprocessed dataset
        for subject_idx, subject_id in enumerate(subjects): 
            pp_epoch_dataset[subject_idx].save(pp_save_dir_path + f"/sub{subject_id}_epo.fif", overwrite=True, verbose = False)
    else:
        # Load preprocessed dataset
        for subject_idx, subject_id in enumerate(subjects): 
            pp_epoch_dataset.append(mne.read_epochs(pp_save_dir_path + f"/sub{subject_id}_epo.fif", preload=True, verbose = False))
    
    # Extract features
    # if features == None:

    #     return 
    
    fscalar_save_dir_path = os.path.join(pp_save_dir_path, 'features/scalar')
    os.makedirs(fscalar_save_dir_path, exist_ok=True)
    
    
    feature_scalar_set = []

    feature_image_set = []

    for ft in features:
        ft_cached = False
        if ft in fe.SCALAR_FEATURE_EXTRACTOR.keys():
            # Check feature cache
            ft_save_file_path = os.path.join(fscalar_save_dir_path, f'{ft}.pkl')
            if os.path.isfile(ft_save_file_path):
                if f_overwrite == True:
                    os.remove(ft_save_file_path)
                    ft_cached = False
                else:
                    ft_cached = True
            
            if ft_cached == False:
                extracted_ftset = fe.extract_scalar_feature_set([ft], pp_epoch_dataset, pp_metadata)
                feature_scalar_set.append(extracted_ftset)

                #TODO: Improve feature set structure before saving

                with open(ft_save_file_path, "wb") as f:
                    pickle.dump(extracted_ftset, f)


            else:
                with open(ft_save_file_path, "rb") as f:
                    extracted_ftset = pickle.load(f)
                    feature_scalar_set.append(extracted_ftset)
                

    
        elif ft in fe.IMAGE_FEATURE_EXTRACTOR.keys():
            #TODO: Implement image feature extraction


            pass
        

    # split train/test dataset
    scalar_train_set = {
        'labels': [],
        'dataset': [],
    }
        
    scalar_test_set = {
        'labels': [],
        'dataset': [],
    }

    if split_type == 'combined':
        split_ratio = kwargs.get('split_ratio', None)

        if split_ratio == None:
            raise ValueError("split_ratio is not specified.")
        if not isinstance(split_ratio, float):
            raise TypeError("split_ratio must be of type float")
        
        for ftset in feature_scalar_set:
            ft_sub_train_dataset = []
            ft_sub_train_labelset = []
            ft_sub_test_dataset = []
            ft_sub_test_labelset = []

            for subj_data in ftset:
                sub_dataset = []
                sub_labelset = []

                for cls_idx, cls_data in enumerate(subj_data):
                    for epoch in cls_data:
                        sub_dataset.append(epoch)
                        sub_labelset.append(cls_idx)
            
                sub_labelset = np.array(sub_labelset)
                sub_dataset = np.array(sub_dataset)
                X_train, X_test, y_train, y_test = train_test_split(sub_dataset, sub_labelset, test_size=split_ratio, random_state=42)

                ft_sub_train_dataset.append(X_train)
                ft_sub_train_labelset.append(y_train)

                ft_sub_test_dataset.append(X_test) 
                ft_sub_test_labelset.append(y_test)
            
            scalar_train_set['dataset'].append(np.concatenate(ft_sub_train_dataset, axis=0))
            scalar_train_set['labels'].append(np.concatenate(ft_sub_train_labelset, axis=0))
            scalar_test_set['dataset'].append(np.concatenate(ft_sub_test_dataset, axis=0))
            scalar_test_set['labels'].append(np.concatenate(ft_sub_test_labelset, axis=0))
        
        #TODO: Image feature type
        

    elif split_type == 'lnso':
        split_test_subjects = kwargs.get('split_test_subjects', None)
        if split_test_subjects == None:
            raise ValueError("test_subjects is not specified.")
        
        # Handle scalar feature type
        for ftset in feature_scalar_set:
            ft_sub_train_dataset = []
            ft_sub_train_labelset = []
            ft_sub_test_dataset = []
            ft_sub_test_labelset = []

            for subj_idx, subj_data in enumerate(ftset):
                sub_dataset = []
                sub_labelset = []

                for cls_idx, cls_data in enumerate(subj_data):
                    for epoch in cls_data:
                        sub_dataset.append(epoch)
                        sub_labelset.append(cls_idx)
            
                sub_labelset = np.array(sub_labelset)
                sub_dataset = np.array(sub_dataset)

                if subjects[subj_idx] in split_test_subjects:
                    ft_sub_test_dataset.append(sub_dataset)
                    ft_sub_test_labelset.append(sub_labelset)
                else:
                    ft_sub_train_dataset.append(sub_dataset)
                    ft_sub_train_labelset.append(sub_labelset)
            
            scalar_train_set['dataset'].append(np.concatenate(ft_sub_train_dataset, axis=0))
            scalar_train_set['labels'].append(np.concatenate(ft_sub_train_labelset, axis=0))
            scalar_test_set['dataset'].append(np.concatenate(ft_sub_test_dataset, axis=0))
            scalar_test_set['labels'].append(np.concatenate(ft_sub_test_labelset, axis=0))
        
        #TODO: Image feature type
    
    else:
        raise ValueError("split_type is not valid.")
    
    # Merge feature type
    scalar_train_set['dataset'] = np.concatenate(scalar_train_set['dataset'], axis=2)
    scalar_train_set['labels'] = scalar_train_set['labels'][0]
    scalar_test_set['dataset'] = np.concatenate(scalar_test_set['dataset'], axis=2)
    scalar_test_set['labels'] = scalar_test_set['labels'][0]

    #TODO: Image feature type

    # Create training folder
    scalar_set_saving_path = os.path.join(v.SAM40_DIR_PREPROCESSED, 'scalar_train_test')
    image_set_saving_path = os.path.join(v.SAM40_DIR_PREPROCESSED, 'image_train_test')

    if os.path.isdir(scalar_set_saving_path):
        shutil.rmtree(scalar_set_saving_path)
    if os.path.isdir(image_set_saving_path):
        shutil.rmtree(image_set_saving_path)

    os.makedirs(scalar_set_saving_path, exist_ok=True)
    os.makedirs(image_set_saving_path, exist_ok=True)

    # Save processed scalar dataset
    with open(os.path.join(scalar_set_saving_path, 'train.pkl'), 'wb') as f:
        pickle.dump(scalar_train_set, f)
    with open(os.path.join(scalar_set_saving_path, 'test.pkl'), 'wb') as f:
        pickle.dump(scalar_test_set, f)
    
    #TODO: Save processed image dataset

    return scalar_set_saving_path, image_set_saving_path
        







    
        

