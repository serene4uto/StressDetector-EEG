import numpy as np

DTS_DIR_ORIGIN = "data/origin"
DTS_DIR_ORIGIN_RAW = DTS_DIR_ORIGIN + "/raw_data"
DTS_ORIGIN_LABEL_PATH = DTS_DIR_ORIGIN + "/scales.xls"

DTS_DIR_PROCESSED = "data/processed"

DTS_SAMPLING_FREQ = 128

DTS_TASK_LIST= ["Relax", "Arithmetic", "Mirror_image", "Stroop"]

DTS_STRESS_LVL_THRESHOLD = 5
DTS_MAX_SUBJECT = 40
DTS_MAX_TRIAL = 3
DTS_MAX_TASK_DURATION_SEC = 25

DTS_SUBJECT_LIST = np.arange(1, DTS_MAX_SUBJECT+1).tolist()
DTS_TRIAL_LIST = np.arange(1, DTS_MAX_TRIAL+1).tolist()

DTS_CLASS_LIST = [
    "Relax",
    "Low Stress",
    "High Stress",
]

DTS_CHANNEL_LIST = [
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

COLUMNS_TO_RENAME = {
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
