import mne
import pickle
import matplotlib.pyplot as plt

Chosen_set_dir = "data/processed/25.0sec_overlap0.5_fp2"

# read metadata
loaded_metadata = None
with open(Chosen_set_dir + "/metadata.pkl", 'rb') as file:
    loaded_metadata = pickle.load(file)

dataset = []
for subject_idx, subject_id in enumerate(loaded_metadata['subjects']):
    dataset.append(mne.read_epochs(Chosen_set_dir + f'/sub{subject_id}_epo.fif', verbose = False))

# fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 6))  # Adjust figure size as needed
# fig.set_dpi(100)
# event_colors = dict({'Relax': 'red', 'Low Stress': 'blue', 'High Stress': 'green'})
event_colors = dict({1: 'red', 2: 'blue', 3: 'green'})

for subj_idx, subj_id in enumerate(loaded_metadata['subjects']):
    dataset[subj_idx].plot(scalings = 100, event_color=event_colors)
# dataset[0]["Low Stress"].compute_psd().plot_topomap()
# dataset[0].plot_sensors(kind="3d", ch_type="all")
plt.show()
