import numpy as np

SAM40_URL = "https://figshare.com/ndownloader/files/27956376"

# SAM40 dataset paths
SAM40_DIR_ORIGIN = ".data/SAM40"
SAM40_DIR_ORIGIN_RAW = ".data/SAM40/raw"
SAM40_ORIGIN_LABEL_PATH = ".data/SAM40/raw/scales.xls"
SAM40_DIR_PREPROCESSED = ".data/SAM40/processed"  

SAM40_SAMPLING_FREQ_HZ = 128

SAM40_TASK_LIST = ["Relax", "Arithmetic", "Mirror_image", "Stroop"]

SAM40_STRESS_LVL_THRESHOLD = 5
SAM40_MAX_SUBJECT = 40
SAM40_MAX_TRIAL = 3
SAM40_MAX_TASK_DURATION_SEC = 25

SAM40_SUBJECT_LIST = np.arange(1, SAM40_MAX_SUBJECT+1).tolist()
SAM40_TRIAL_LIST = np.arange(1, SAM40_MAX_TRIAL+1).tolist()

SAM40_CLASS_LIST = [
    "Relax",
    "Low Stress",
    "High Stress",
]

SAM40_CHANNEL_LIST = [
 'Cz',
 'Fz',
 'Fp1',
 'F7',
 'F3',
 'FC1',
 'C3',
 'FC5',
 'FT9',
 'T7',
 'CP5',
 'CP1',
 'P3',
 'P7',
 'PO9',
 'O1',
 'Pz',
 'Oz',
 'O2',
 'PO10',
 'P8',
 'P4',
 'CP2',
 'CP6',
 'T8',
 'FT10',
 'FC6',
 'C4',
 'FC2',
 'F4',
 'F8',
 'Fp2']

SAM40_COLUMNS_TO_RENAME = {
    'Subject No.': 'subject_no',
    'Trial_1'   : 't1_Arithmetic',
    'Unnamed: 2': 't1_Mirror_image',
    'Unnamed: 3': 't1_Stroop',
    'Trial_2'   : 't2_Arithmetic',
    'Unnamed: 5': 't2_Mirror_image',
    'Unnamed: 6': 't2_Stroop',
    'Trial_3'   : 't3_Arithmetic',
    'Unnamed: 8': 't3_Mirror_image',
    'Unnamed: 9': 't3_Stroop'
}