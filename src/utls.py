from scipy.signal import butter, filtfilt, welch
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates filter coefficients for bandpass butterworth filter.

    Args:
        lowcut (int): Lower frequency edge
        highcut (int): Upper frequency edge
        fs (int): Sampling rate
        order (int, optional): Filter order. Defaults to 5.

    Returns:
        (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the butter filter:
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Filter data using a bandpass butterworth filter.

    Args:
        data (ndarray): Raw data
        lowcut (int): Lower frequency edge
        highcut (int): Upper frequency edge
        fs (int): Sampling rate
        order (int, optional): _description_. Defaults to 5.

    Returns:
        ndarray: Filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def prep_raw_data(data, fs, lowcut = 1, highcut = 15):
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

    idx_freq_oi = np.logical_and(freqs > 1, freqs < 11)

    return data_zm, freqs[idx_freq_oi], psd[idx_freq_oi]

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
