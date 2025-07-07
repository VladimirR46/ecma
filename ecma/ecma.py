"""
@author: Antipov Vladimir, Badarin Artem
"""

import numpy as np
from sklearn.cluster import KMeans
from sacfit import saccade_fit_xy
from sklearn.metrics import r2_score

def add_delays(signal: np.ndarray, n_delay: int) -> np.ndarray:
    """
    Adds time delays to the input signal.

    Args:
        signal (np.ndarray): Input signal, 1D or 2D array.
        n_delay (int): Number of time delays.

    Returns:
        np.ndarray: Delayed feature matrix.
    """
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    n_samples, n_features = signal.shape

    if n_delay < 0:
        raise ValueError("The parameter n_delay cannot be negative.")
    if n_delay >= n_samples:
        raise ValueError("The parameter n_delay must be less than the number of time samples (n_samples).")

    slices = [
        signal[shift: shift + (n_samples - n_delay), :]
        for shift in range(n_delay + 1)
    ]
    delayed_matrix = np.hstack(slices)
    return delayed_matrix


def cluster_weighting(xpos: np.ndarray, ypos: np.ndarray, n_delay: int = 5,
                      window_time: float = 0.2, step_time: float = 0.02, freq: int = 500) -> np.ndarray:
    """
    Calculates a cluster-based weighting mask for eye movement data using sliding
    k-means clustering with time-delay embedding.

    Args:
      xpos (np.ndarray): Horizontal gaze positions.
      ypos (np.ndarray): Vertical gaze positions.
      n_delay (int, optional): Number of time delays to apply. Default is 5.
      window_time (float, optional): Sliding window length in seconds. Default is 0.2.
      step_time (float, optional): Step size for window movement in seconds. Default is 0.02.
      freq (int, optional): Sampling frequency in Hz. Default is 500.

    Returns:
      np.ndarray: Weighting mask of the same length as input signals, where detected
                  clustered segments are marked with 1, and others with 0.
    """
    stack_data = np.stack([xpos, ypos], axis=1)
    data = add_delays(stack_data, n_delay=n_delay)
    num_samples = int(window_time / (1. / freq))
    step_size = np.max([1, int(step_time / (1. / freq))])

    num_tests = np.zeros(len(xpos))
    total_weights = np.zeros(len(xpos))
    for i in range(0, len(data) - num_samples + 1, step_size):
        idx = range(i, i + num_samples)
        window_data = data[idx]
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=300)
        labels = kmeans.fit_predict(window_data)
        switches = np.abs(np.diff(labels))
        if np.sum(switches) == 1:
            fi = np.where(switches == 1)[0][0]
            ei = min(fi + n_delay * 0 + 0, len(switches))
            switches[fi - 0 - n_delay * 0:ei] = 1
            # switchesw = 1./ np.sum(switches)
            weighted = np.hstack([switches, 0])
            total_weights[idx] = total_weights[idx] + weighted
            num_tests[idx] = num_tests[idx] + 1

    # total_weights = total_weights/num_tests
    mask = num_tests > 0
    total_weights[mask] = total_weights[mask] / num_tests[mask]
    total_weights[total_weights > 0] = 1
    return total_weights


def win_approx(total_weights, xpos, ypos, freq=500, offset_time=0.05):
    t = 1000 * np.arange(0, len(xpos)) / freq
    diff = np.diff(total_weights)
    start_idx = np.where(diff == 1)[0] + 1
    end_idx = np.where(diff == -1)[0]
    if total_weights[-1] == 1 and len(start_idx) > len(end_idx):
        start_idx = start_idx[:-1]

    offset = int(offset_time / (1. / freq))
    begin_peaks, end_peaks, durations, dist, peaks = [], [], [], [], []
    r2_x, r2_y = [], []
    for i in range(len(start_idx)):
        peak = (start_idx[i] + end_idx[i]) // 2
        if i == 0:
            left_border = max(0, peak - offset)
        else:
            left_border = max(peak - offset, (end_idx[i - 1] + start_idx[i]) // 2)

        if i == len(start_idx) - 1:
            right_border = min(peak + offset, len(xpos))
        else:
            right_border = min(peak + offset, (start_idx[i + 1] + end_idx[i]) // 2)

        idx = range(left_border, right_border)
        win_x, win_y = xpos[idx], ypos[idx]
        fit_data = saccade_fit_xy(np.array(win_x), np.array(win_y), freq)
        if fit_data is not None and fit_data['dur'] >= 0.01 and fit_data['amp'] > 40:
            begin_peaks.append(idx[0] + fit_data['begin_peak'])
            end_peaks.append(idx[0] + fit_data['end_peak'])
            peaks.append((end_peaks[-1] + begin_peaks[-1]) // 2)
            durations.append(fit_data['dur'])
            dist.append(fit_data['dist'])
            r2_x.append(r2_score(win_x, fit_data['sac_x']))
            r2_y.append(r2_score(win_y, fit_data['sac_y']))

    info = {
        'dur': durations,
        'peaks': peaks,
        'start': begin_peaks,
        'end': end_peaks,
        'startT': t[begin_peaks],
        'endT': t[end_peaks],
        'dist': dist,
        'r2_x': r2_x,
        'r2_y': r2_y,
    }
    return info


def ecma(xpos, ypos):
    total_weights = cluster_weighting(xpos, ypos)

    sac_info = win_approx(total_weights, xpos, ypos, freq=500, offset_time=0.3)

    return sac_info




