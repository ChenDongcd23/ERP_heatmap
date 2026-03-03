'''ERP_heatmap'''

from preprocessing_utils import *
from heatmap_utils import *
from get_peak import find_local_peak
from matplotlib.colors import LinearSegmentedColormap


import mne
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


## 0.Read data =======================================================

clean_file = r'D:/test.fif'
clean_data = mne.io.read_raw_fif(clean_file, preload=True)
#clean_data.plot(block=True)


## 1. Preparation Epochs ==============================================

# classify the labels
events, event_id = mne.events_from_annotations(clean_data)

print(f'[INFO] Event ID:', event_id)
print(f'[TEST] Events:', events.shape)

stim_labels = {'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3}
resp_labels = {'Stimulus/S  4': 4, 'Stimulus/S  5': 5}

stim_events = mne.pick_events(events, include = [1, 2, 3])
resp_events = mne.pick_events(events, include = [4, 5])

# reaction time 
stim_onsets, resp_onsets = get_onsets(clean_data, stim_labels, resp_labels)
reaction_time_s = (resp_onsets - stim_onsets) * 1000

# metadata
metadata_ori = pd.DataFrame({
    'stim': stim_events[:,2],
    'resp': resp_events[:,2],
    'rtime': reaction_time_s,
#   'peak_amp': np.nan,
#   'peak_latency': np.nan
    })

# set epochs
epochs = mne.Epochs(
    raw = clean_data,
    events = resp_events,
    event_id = resp_labels,
    tmin = -1.0,
    tmax =  1.0,
    baseline = (-0.5,-0.3),
    picks = ['FC3', 'C3', 'CP3'],
    metadata= metadata_ori,
    preload = True)

# drop bad epochs
reject_criteria = dict(eeg=100e-6)
epochs.drop_bad(reject=reject_criteria)
print(f'[TEST] shape', epochs.get_data().shape)

# fill the metadata
kept_id = epochs.selection
metadata_upd = metadata_ori.copy()
metadata_upd['drop_log'] = True
metadata_upd.loc[kept_id, 'drop_log'] = False

#show(metadata_upd)


## 2. Heatmap ===================================================

epochs_valid = epochs["resp == 4 and rtime <= 1500"]

# stim_labels = {'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3}
# resp_labels = {'Stimulus/S  4': 4, 'Stimulus/S  5': 5}

# stim = {'Congruence': 1, 'Neutral': 2, 'Incongruence': 3}
# resp = {'Correct': 4, 'Incorrect': 5}

# filters = {'stim': [1,2,3], 'resp': [1,2]}


erp_heatmap(
        epochs_valid, 
        labels = None, 
        picks = 'all',
        sort_by_rt = False,
        smooth_win = 5, 
        track_line = False)


