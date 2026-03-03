"""heatmap functions"""


import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns


# Epochs slicing fuction ======================================================

def _filter_epochs(epochs, filters=None):
    
    '''
    Example:
    
    filters = {"stim": [1,2], "resp": 1}
    '''
    
    if filters is None:
        return epochs
    
    meta = epochs.metadata.copy()
    mask = np.ones(len(meta), dtype=bool)
    
    for key, value in filters.items():
        
        if key not in meta.columns:
            raise ValueError(f"{key} not in metadata")
            
        if not isinstance(value, (list, tuple, np.ndarray)):
            value = [value]
        
        mask &= meta[key].isin(value)
            
    return epochs[mask]


# Epoch data extraction =======================================================

def _epoch_filter(
        epochs, 
        filters=None, 
        picks=None, 
        average=False
        ):
    
    # filter epochs
    filtered_epochs = _filter_epochs(epochs, filters)
    filtered_meta = filtered_epochs.metadata
    
    if len(filtered_epochs) == 0:
        raise ValueError("NO trials left after filtering.")
    
    # extract data
    datx = filtered_epochs.get_data(picks=picks, units=dict(eeg="uV"))
    timx = filtered_epochs.times
    
    if picks is None:
        ch_names = filtered_epochs.ch_names
    else:
        if isinstance(picks, str):
            picks_list = [picks]
        else:
            picks_list = picks
        ch_names = [
           filtered_epochs.ch_names[p] if isinstance(p, int) else p
           for p in picks_list
        ]
    
    if average:
        datx = np.mean(datx, axis=1, keepdims=True)
        ch_names = ["Ave"]
        
    n_trials, n_channels, n_times = datx.shape
        
    dfs = {}
    for ch_idx, ch_name in enumerate(ch_names):
        
        datum = datx[:, ch_idx, :]
        
        df = pd.DataFrame(
            datum,
            index = filtered_meta.index, # trial_idx
            columns = np.array(timx, dtype=float),  # time samples
        )
        
        dfs[ch_name] = df
        
    return dfs, filtered_meta
        

# Resort trials ===============================================================

def _sort_by_rt(df, meta_df, ascending=False):
    
    sorted_trials = meta_df["rtime"].sort_values(
        ascending = ascending
    ).index
    
    df_sorted = df.loc[sorted_trials]
    meta_sorted = meta_df.loc[sorted_trials]
    
    return df_sorted, meta_sorted
 
       
# Smooth trials ===============================================================   
def _smooth_trials(df, meta_df, win=10, center=True):

    df = df.astype(float)
    
    df_smooth = df.rolling(
        window = win,
        center = center,
        min_periods = win
    ).mean().dropna()
    
    meta_smooth = meta_df.loc[df_smooth.index]
    
    return df_smooth, meta_smooth
        

# transformation ================================================================

# stim_labels = {'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3}
# resp_labels = {'Stimulus/S  4': 4, 'Stimulus/S  5': 5}

# stim = {'Congruence': 1, 'Neutral': 2, 'Incongruence': 3}
# resp = {'Correct': 4, 'Incorrect': 5}

# filters = {'stim': [1,2,3], 'resp': [1,2]}

def _trans_dict(labels):
    
    '''
    Parameters
    ----------
    labels : str or list of str
    eg. 'Congruence' / 'Correct' or ['Congruence','Neutral']
    
    Returns
    -------
    filter_dict : dict
        {'stim':1} / {'stim':[1,3]} / {'resp':4} / {'resp':[4,5]}
    labels_list : list
    '''
    
    stim = {'Congruence': 1, 'Neutral': 2, 'Incongruence': 3}
    stim_labels = {'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3}
    resp = {'Correct': 4, 'Incorrect': 5}
    resp_labels = {'Stimulus/S  4': 4, 'Stimulus/S  5': 5}
 
    if isinstance(labels, str):
        labels_list = [labels]
    elif isinstance(labels, (list, tuple)):
        labels_list = list(labels)
    else:
        raise ValueError("labels must be str or list/tuple of str")
        
    filter_dict = {}
    #labels_dict = {'stim': [], 'resp': []}

    for lb in labels_list:
        if lb in stim:
            kind = "stim"
            num = stim[lb]
            # metadata value
            found = False
            for k, v in stim_labels.items():
                if v == num:
                    filter_dict.setdefault(kind, []).append(v)
                    found = True
                    break
            if not found:
                raise ValueError(f"Label '{lb}' not found in stim metadata map.")
            #labels_dict[kind].append(lb)

        elif lb in resp:
            kind = "resp"
            num = resp[lb]
            # metadata value
            found = False
            for k, v in resp_labels.items():
                if v == num:
                    filter_dict.setdefault(kind, []).append(v)
                    found = True
                    break
            if not found:
                raise ValueError(f"Label '{lb}' not found in resp metadata map.")
            #labels_dict[kind].append(lb)

        else:
            raise ValueError(f"Label '{lb}' not recognized in stim or resp.")

    for k in filter_dict:
        if len(filter_dict[k]) == 1:
            filter_dict[k] = filter_dict[k][0]

    return filter_dict
        

# process =====================================================================
def process_epochs(
        epochs, 
        labels = None,
        picks = None,
        sort_by_rt = False,
        smooth_win = None
        ):
    
    # filtered epochs
    if labels is None:
        filter_dict = None
    else:
        filter_dict = _trans_dict(labels)

    # select channels
    if picks is None or picks == 'all' or (isinstance(picks, (list, tuple)) and len(picks) > 1):
        average = True
    else:
        average = False        
        
    dfs, meta_df = _epoch_filter(
        epochs = epochs, 
        filters = filter_dict, 
        picks = picks, 
        average = average)

    if average:
        df_f = dfs['Ave']
        
        if picks is None or picks == 'all':
            ch_names = epochs.ch_names
        else:
            ch_names = picks if isinstance(picks, list) else [picks]
        ch_str = f"Ave ({','.join(ch_names)})"
        
    else:
        ch_name = picks if isinstance(picks, str) else picks[0]
        df_f = dfs[ch_name]
        ch_str = ch_name
    
    if sort_by_rt:
        df_f, meta_df = _sort_by_rt(df_f, meta_df, ascending=False)
        
    if smooth_win is not None:
        df_f, meta_df = _smooth_trials(df_f, meta_df, win=smooth_win, center=True)

    return df_f, meta_df, ch_str
    
       
# Heatmap =====================================================================

# plot

def heatmap(df, meta_df, track_line=True):

    line_data = df.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(8,6),
        sharex=True,
        gridspec_kw={'height_ratios':[4,1], 'hspace':0.02})
    
    plt.subplots_adjust(bottom=0.2)
    
    cmap = LinearSegmentedColormap.from_list(
        "eeg_cmap", 
        ["blue","cyan","yellow","red"])
    
    im = sns.heatmap(df, cmap=cmap, ax=ax1, cbar=False)
    
    ax1.invert_yaxis()
    n_trials = df.shape[0]
    n_time = df.shape[1]
    
    time_values = np.linspace(-1000, 1000, n_time)
    zero_idx = np.argmin(np.abs(time_values)) + 0.5
    
    yticks_idx = np.arange(0, n_trials, 20)
    yticks_labels = [str(1 if i == 0 else i) for i in yticks_idx]
    if yticks_idx[-1] != n_trials - 1:
        yticks_idx = np.append(yticks_idx, n_trials - 1)
        yticks_labels.append(str(n_trials))
    
    ax1.set_yticks(yticks_idx)
    ax1.set_yticklabels(yticks_labels, rotation=0)
    ax1.tick_params(axis='x', bottom=False, labelbottom=False)
    ax1.set_ylabel('Trial', fontsize=12)
    ax1.set_title('Heatmap', fontsize=14, pad=12)
    ax1.axvline(x=zero_idx, color='black', linestyle='--', linewidth=2)
    
    if track_line:
        stim_onset = 0 - meta_df["rtime"].values
        t_min = time_values[0]
        t_max = time_values[-1]
        
        stim_idx = (stim_onset - t_min) / (t_max - t_min) * (n_time - 1)
        x_plot = stim_idx + 0.5
        y_plot = np.arange(n_trials) + 0.5
        ax1.plot(x_plot, y_plot, color='black', linewidth=2)
    
    pos1 = ax1.get_position()
    cbar_width = 0.05
    cbar_ax = fig.add_axes([pos1.x1 + 0.03, pos1.y0, cbar_width, pos1.height])
    cbar = fig.colorbar(im.get_children()[0], cax=cbar_ax)
    vmin = df.values.min()
    vmax = df.values.max()
    cticks = np.arange(int(vmin), int(vmax), 5)
    cbar.set_ticks(cticks)
    
    x = np.arange(n_time) + 0.5
    y = line_data.values
    sns.lineplot(x=x, y=y, ax=ax2, color='blue')
    
    ax2.axvline(x=zero_idx, color='black', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    xticks = np.arange(-1000, 1000+1, 500)
    tick_idx = [np.argmin(np.abs(time_values - t)) for t in xticks]
    ax2.set_xticks(np.array(tick_idx) + 0.5)
    ax2.set_xticklabels(xticks)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('μV')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    plt.show()        
        


    
def erp_heatmap(
        epochs, 
        labels = 'Incongruence', 
        picks = 'C3',
        sort_by_rt = False,
        smooth_win = None, 
        track_line = True):
    
    
    df_f, meta_df, ch_str = process_epochs(
        epochs, 
        labels = labels,
        picks = picks,
        sort_by_rt = sort_by_rt,
        smooth_win = smooth_win)
    
    if labels is None:
        labels = 'All'

    line_data = df_f.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(8,6),
        sharex=True,
        gridspec_kw={'height_ratios':[4,1], 'hspace':0.02})
    
    plt.subplots_adjust(bottom=0.2)
    
    cmap = plt.get_cmap('jet')
    im = sns.heatmap(df_f, cmap=cmap, ax=ax1, cbar=False)
    
    ax1.invert_yaxis()
    n_trials = df_f.shape[0]
    n_time = df_f.shape[1]
    
    time_values = np.linspace(-1000, 1000, n_time)
    zero_idx = np.argmin(np.abs(time_values)) + 0.5
    
    yticks_idx = np.arange(0, n_trials, 50)
    yticks_labels = [str(1 if i == 0 else i) for i in yticks_idx]
    if yticks_idx[-1] != n_trials - 1:
        yticks_idx = np.append(yticks_idx, n_trials - 1)
        yticks_labels.append(str(n_trials))
    
    ax1.set_yticks(yticks_idx)
    ax1.set_yticklabels(yticks_labels, rotation=0)
    ax1.tick_params(axis='x', bottom=False, labelbottom=False)
    ax1.set_ylabel('Trial', fontsize=12)
    ax1.set_title(f'{labels}-trials Heatmap: {ch_str}    [smooth_win={smooth_win}]', 
                  fontsize=14, pad=12)
    ax1.axvline(x=zero_idx, color='black', linestyle='--', linewidth=2)
    
    if track_line:
        stim_onset = 0 - meta_df["rtime"].values
        t_min = time_values[0]
        t_max = time_values[-1]
        
        stim_idx = (stim_onset - t_min) / (t_max - t_min) * (n_time - 1)
        x_plot = stim_idx + 0.5
        y_plot = np.arange(n_trials) + 0.5
        ax1.plot(x_plot, y_plot, color='black', linewidth=2)
    
    pos1 = ax1.get_position()
    cbar_width = 0.05
    cbar_ax = fig.add_axes([pos1.x1 + 0.03, pos1.y0, cbar_width, pos1.height])
    cbar = fig.colorbar(im.get_children()[0], cax=cbar_ax)
    vmin = df_f.values.min()
    vmax = df_f.values.max()
    cticks = np.arange(int(vmin), int(vmax), 5)
    cbar.set_ticks(cticks)
    
    x = np.arange(n_time) + 0.5
    y = line_data.values
    sns.lineplot(x=x, y=y, ax=ax2, color='blue')
    
    ax2.axvline(x=zero_idx, color='black', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    xticks = np.arange(-1000, 1000+1, 500)
    tick_idx = [np.argmin(np.abs(time_values - t)) for t in xticks]
    ax2.set_xticks(np.array(tick_idx) + 0.5)
    ax2.set_xticklabels(xticks)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('μV')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    plt.show()    
        
        
        
        
        
        
        
        