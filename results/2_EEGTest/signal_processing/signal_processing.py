import numpy as np
from scipy import signal

FS = 250
NYQUIST = .5 * FS

LOW_FREQS = range(4, 41, 4)
QUANTITY_BANDS = len(LOW_FREQS)-1


def filter_in_all_frequency_bands(trial):
    filtered_signals = np.zeros((QUANTITY_BANDS, *trial.shape))
    for n_low_freq in range(0, len(LOW_FREQS)):
        low_freq = LOW_FREQS[n_low_freq]
        if low_freq == LOW_FREQS[-1]:
            break

        high_freq = LOW_FREQS[n_low_freq+1]

        # Create a 5 order Chebyshev Type 2 filter to the specific band (low_freq - high_freq)
        b, a = signal.cheby2(5, 0.5, [low_freq, high_freq], btype="bandpass", fs=250)

        filtered_signals[n_low_freq, :, :] = signal.filtfilt(b, a, trial, axis=0)

    return filtered_signals


def filter_bank(eeg):
    filtered_signals = np.zeros((QUANTITY_BANDS, *eeg.shape))
    for n_trial in range(eeg.shape[0]):
        trial = eeg[n_trial, :, :]
        filtered_signals[:, n_trial, :, :] = filter_in_all_frequency_bands(trial)

    return filtered_signals


def apply_bandpass_filter(eeg):
    eeg.left_data = bandpass_filter(eeg.left_data)
    eeg.right_data = bandpass_filter(eeg.right_data)


def bandpass_filter(eeg_signal):
    b, a = signal.butter(4, [8, 30], btype="bandpass", fs=250)
    return signal.filtfilt(b, a, eeg_signal, axis=1)
