import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt, welch, spectrogram
from pathlib import Path
from sklearn.decomposition import PCA

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Creates filter coefficients for bandpass butterworth filter.

    Args:
        lowcut (int): Lower frequency edge
        highcut (int): Upper frequency edge
        fs (int): Sampling rate
        order (int, optional): Filterorder. Defaults to 4.

    Returns:
        (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the butter filter:
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_highpass(lowcut, fs, order=4):
    """Creates filter coefficients for bandpass butterworth filter.

    Args:
        lowcut (int): Lower frequency edge
        highcut (int): Upper frequency edge
        fs (int): Sampling rate
        order (int, optional): Filterorder. Defaults to 4.

    Returns:
        (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the butter filter:
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low,  btype='high')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filter data using a bandpass butterworth filter.

    Args:
        data (ndarray): Raw data
        lowcut (int): Lower frequency edge
        highcut (int): Upper frequency edge
        fs (int): Sampling rate
        order (int, optional): Filderorder. Defaults to 4.

    Returns:
        ndarray: Filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, lowcut, fs, order=4):
    """Filter data using a highpass butterworth filter.

    Args:
        data (ndarray): Raw data
        lowcut (int): Lower frequency edge
        fs (int): Sampling rate
        order (int, optional): Filterorder. Defaults to 4.

    Returns:
        ndarray: Filtered data
    """
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def prep_raw_data(data, fs, lowcut = 1, highcut = 12):
    """Filter and zero-mean data.

    Args:
        data (ndarray): Raw data
        fs (int): Sampling rate
        lowcut (int, optional): Lower frequency edge. Defaults to 1.
        highcut (int, optional): Upper frequency edge. Defaults to 15.

    Returns:
        ndarray, ndarray, ndarray : Filtered zero mean data, psd and frqs estiamted with welch method.
    """
    # process data in time domain
    filt = butter_bandpass_filter(data, lowcut, highcut, fs)
    data_zm = filt - np.nanmean(filt)

    # process data in freq domain
    freqs, psd = welch(data_zm, fs, nperseg = len(data_zm))

    idx_freq_oi = np.logical_and(freqs > lowcut, freqs < highcut)

    return data_zm, freqs[idx_freq_oi], psd[idx_freq_oi]

def prep_raw_data_pca(data, fs, lowcut = 0.1):
    """HP Filter (0.1Hz) PCA to obtain main movement direction.

    Args:
        data (ndarray): Raw data
        fs (int): Sampling rate

    Returns:
        ndarray : Filtered zero mean data, psd and frqs estiamted with welch method.
    """
    # process data in time domain
    raw = np.array(data).T
    filt = butter_highpass_filter(raw, 0.1, fs)

    pca = PCA(n_components = 3)
    pcs = np.empty((max(raw.shape), 3))

    # pca first hand
    pcs[:, :] = pca.fit_transform(filt.T)


    # get first spec esttimation
    freqs, psd = welch(pcs[:,0], fs, nperseg = fs * 2)

    lower_bound = 3
    upper_bound = 10

    # check if filt boundaries make sense
    if lower_bound < 3:
        lower_bound = 3
    elif upper_bound > 10:
        upper_bound = 10
    elif abs(upper_bound - lower_bound) < 4:
        lower_bound = 3
        upper_bound = 9

        
    # filter again around peak frequency
    filt = butter_bandpass_filter(pcs[:,0], lower_bound, upper_bound, fs)
    data_zm = filt - np.nanmean(filt)

    idx_freq_oi = np.logical_and(freqs > 2, freqs < 10)

    return data_zm, freqs[idx_freq_oi], psd[idx_freq_oi], pca.explained_variance_ratio_

def plot_specs(data_all, fs, freq_lims):
    """_summary_

    Args:
        data_all (_type_): _description_
        fs (_type_): _description_
        freq_lims (_type_): _description_
    """

    nms_axis = ['x','y','z']
    sns.set_palette("viridis")

    for i,d in enumerate(data_all):
        freqs, psd = welch(d, fs, nperseg = fs, average='median')
        sns.lineplot(freqs, psd, label=nms_axis[i])   

    plt.xlim(freq_lims) 
    plt.legend()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

def compute_repeated_stats(data, task_oi):

    if not data['task'].str.contains(task_oi).any():
        raise ValueError("Taskname not in dataset")

    idx_task_oi = data["task"] == task_oi
    idx_visit_oi = data["visit"] != "Follow-up"
    data_task = data[np.logical_and(idx_task_oi,idx_visit_oi)]
    data_task = data_task.groupby('id').filter(lambda x: {"Inital","14-days"}.issubset(x['visit']))
    sumed_axis = data_task.groupby(['id','visit'],as_index=False)[["peak amplitude","peak frequency"]].sum()

    n_rows = sumed_axis.shape[0]
    sumed_axis["Norm amp"] = np.nan
    sumed_axis["Norm freq"] = np.nan

    for i in range(0,n_rows):
        idx_id = sumed_axis["id"].str.match(sumed_axis["id"][i])
        sumed_axis["Norm amp"][i] = sumed_axis["peak amplitude"][i] / np.sum(sumed_axis["peak amplitude"][idx_id])
        sumed_axis["Norm freq"][i] = sumed_axis["peak frequency"][i] / np.sum(sumed_axis["peak frequency"][idx_id])

    t_amp, p_amp = stats.ttest_rel(sumed_axis["Norm amp"][sumed_axis.visit == "Inital"], sumed_axis["Norm amp"][sumed_axis.visit == "14-days"])
    t_freq, p_freq = stats.ttest_rel(sumed_axis["Norm freq"][sumed_axis.visit == "Inital"], sumed_axis["Norm freq"][sumed_axis.visit == "14-days"])
    
    return t_amp, p_amp, t_freq, p_freq


def save_raw_tf_plt(x_zm, y_zm, z_zm, cfg_srate, cfg_freqs_oi, id, path_out):
    """_summary_

    Args:
        x_zm (_type_): _description_
        y_zm (_type_): _description_
        z_zm (_type_): _description_
        cfg_srate (_type_): _description_
        cfg_freqs_oi (_type_): _description_
        id (_type_): _description_
        path_out (_type_): _description_
    """

    freqs, times, specs_x = spectrogram(x_zm, cfg_srate, nperseg = cfg_srate, noverlap = cfg_srate // 2)
    freqs, times, specs_y = spectrogram(y_zm, cfg_srate, nperseg = cfg_srate, noverlap = cfg_srate // 2)
    freqs, times, specs_z = spectrogram(z_zm, cfg_srate, nperseg = cfg_srate, noverlap = cfg_srate // 2)
    fig,axs = plt.subplots(2,1)

    time_vec = np.linspace(0,len(x_zm) / cfg_srate, len(x_zm))
    axs[0].plot(time_vec,x_zm,label='x')
    axs[0].plot(time_vec,y_zm,label='y')
    axs[0].plot(time_vec,z_zm,label='z')
    axs[0].set_ylabel('Amplitude [a.u.]')
    axs[0].set_xlabel('Time [sec]')
    axs[0].legend()

    axs[1].pcolormesh(times, freqs,np.array([specs_x, specs_y, specs_z]).sum(axis=0), shading='nearest')
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_ylim(cfg_freqs_oi)

    fname = id + 'raw_data.png'
    plt.savefig(Path.joinpath(path_out,fname))
    plt.close()


def save_raw_tf_plt_pca(x, y, z, pca_zm, eigenratios, cfg_srate, cfg_freqs_oi, id, path_out):
    """_summary_

    Args:
        pca_zm (_type_): _description_
        cfg_srate (_type_): _description_
        cfg_freqs_oi (_type_): _description_
        id (_type_): _description_
        path_out (_type_): _description_
    """

    freqs, times, specs_pca = spectrogram(pca_zm, cfg_srate, nperseg = cfg_srate, noverlap = cfg_srate // 2)
    time_vec = np.linspace(0,len(pca_zm) / cfg_srate, len(pca_zm))
    
    fig,axs = plt.subplots(2,1)
    fig.suptitle("Sitting relaxed")

    axs[0].plot(time_vec,pca_zm,label='pca')
    axs[0].plot(time_vec,x,label='x')
    axs[0].plot(time_vec,y,label='y')
    axs[0].plot(time_vec,z,label='z')
    axs[0].set_ylabel('Amplitude [a.u.]')
    axs[0].set_xlabel('Time [sec]')
    axs[0].legend()

    axs[1].pcolormesh(times, freqs,np.array([specs_pca]).sum(axis=0), shading='nearest')
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_ylim(cfg_freqs_oi)
    axs[1].set_title(eigenratios)

    
    fname = id + 'raw_data.png'
    plt.savefig(Path.joinpath(path_out,fname))
    plt.close()