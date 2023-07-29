import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pandas as pd

import abcgan.constants as const

# import importlib
# importlib.reload(util)
def plot_wtec_data(data, **kwargs):
    bins = kwargs.get('bins', 100)
    tid_type = data[list(data.keys())[0]].get("tid_type", "WTEC")
    locations = list(data.keys())
    fig, ax = plt.subplots(len(const.wtec_vars) + 1, 3, figsize=(20, 20))
    ax = ax.flatten()
    for i in range(len(const.wtec_names)):
        hue = []
        for loc in locations:
            hue += [f'{loc}'] * len(data[loc]['utc'])
        sns.histplot(x=np.hstack([data[loc]['wtecs'][:, i] for loc in locations]),
                     bins=bins, ax=ax[i], element='step', hue=hue, alpha=0.5)
        ax[i].set_xlabel(f'{const.wtec_units[i]}', fontsize=15)
        ax[i].set_ylabel('Counts')
        ax[i].set_title(f'{const.wtec_names[i]}', fontsize=20)
    hue = []
    for loc in locations:
        hue += [f'{loc}'] * len(data[loc]['utc'])
    sns.histplot(x=np.hstack([[datetime.datetime.utcfromtimestamp(d) for d in data[loc]['utc']] for loc in locations]),
                 bins=bins, ax=ax[-1], hue=hue, element='step', alpha=0.8)
    ax[-1].set_xlabel(f'Date', fontsize=15)
    ax[-1].set_ylabel('Counts')
    ax[-1].set_title(f'Unix Timestamps', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{tid_type} Waves", fontsize=30)
    plt.show()
    plt.close(fig)


def plot_real_vs_generated_wtec(real,
                                generated,
                                hel_dists=None,
                                hist_info=None,
                                tid_type=const.wtec_default_tid_type,
                                location=const.wtec_default_location,
                                title_annot=None,
                                wtec_var="avg",
                                n_bins=100):
    if hist_info is not None:
        (r_hists, f_hists, edges) = hist_info
        if isinstance(r_hists, np.ndarray):
            r_hists = r_hists.T
            f_hists = f_hists.T
            edges = edges.T
    feat_indexes = np.array([i for i, n in enumerate(const.wtec_names) if n.find(wtec_var) > 0])
    plt.style.use('default')
    pal = sns.color_palette(n_colors=2)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax = ax.flatten()
    for i, w_index in enumerate(feat_indexes):
        x = np.hstack((real[..., w_index].flatten(), generated[..., w_index].flatten()))
        hue = ['Real'] * len(real[..., w_index].flatten()) + ['Generated'] * len(generated[..., w_index].flatten())
        data = pd.DataFrame({f'{const.wtec_units[w_index]}': x, 'Type': hue})
        if hist_info is not None:
            n_bins = len(r_hists[w_index])
        sns.histplot(data=data, x=f'{const.wtec_units[w_index]}', hue='Type', ax=ax[i],
                     element='step', bins=n_bins, legend=True, stat='density',
                     alpha=0.2, common_norm=False, palette=pal,
                     binrange=const.wtec_dict[tid_type]["meas_ranges"][w_index],
                     hue_order=['Generated', 'Real'])
        if hist_info is not None:
            if isinstance(edges, list):
                xlabel = edges[w_index]
            else:
                xlabel = edges[w_index][1:] - np.diff(edges[w_index]) / 2
            ax[i].plot(xlabel, r_hists[w_index], linewidth=2, color=pal[1], label='Actual')
            ax[i].plot(xlabel, f_hists[w_index], linewidth=2, color=pal[0], label='Generated')
        if hel_dists is not None:
            ax[i].set_title(f'{const.wtec_names[w_index]} (Hel Dist = {hel_dists[w_index]:.3f})')
        else:
            ax[i].set_title(f'{const.wtec_names[w_index]}')
    if title_annot is None:
        if hel_dists is not None:
            plt.suptitle(f'{tid_type}_{location} Model Results || Avg = {hel_dists.mean():.3f}', fontsize=20)
        else:
            plt.suptitle(f'{tid_type}_{location} Model Results', fontsize=20)
    else:
        plt.suptitle(f'{title_annot} || Avg = {hel_dists.mean():.3f}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    plt.close(fig)