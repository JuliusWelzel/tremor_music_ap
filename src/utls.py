from scipy.signal import butter, filtfilt, welch
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
    freqs, psd = welch(data_zm, fs, nperseg = fs, average='median')

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

    
