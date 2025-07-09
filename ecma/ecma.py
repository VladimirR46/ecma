"""
@author: Antipov Vladimir, Badarin Artem
"""

import numpy as np
from sklearn.cluster import KMeans
from .sacfit import saccade_fit_xy
from sklearn.metrics import r2_score

def _add_delays(signal: np.ndarray, n_delay: int) -> np.ndarray:
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


def _cluster_weighting(xpos: np.ndarray, ypos: np.ndarray, n_delay: int = 5,
                      window_time: float = 0.1, step_time: float = 0.04, freq: int = 500) -> np.ndarray:
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
    data = _add_delays(stack_data, n_delay=n_delay)
    num_samples = int(window_time / (1. / freq))
    step_size = np.max([1, int(step_time / (1. / freq))])

    total_weights = np.zeros(len(xpos))
    for i in range(0, len(data) - num_samples + 1, step_size):
        idx = range(i, i + num_samples)
        window_data = data[idx]
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=300)
        labels = kmeans.fit_predict(window_data)
        switches = np.abs(np.diff(labels))
        if np.sum(switches) == 1:
            fi = np.where(switches == 1)[0][0]
            total_weights[fi + idx[0] + 1] = 1
    return total_weights


def _win_approx(total_weights, xpos, ypos, freq=500, offset_time=0.05):
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
        if fit_data is not None and fit_data['dur'] >= 0.01 and fit_data['amp'] > 20:
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


def ecma(xpos, ypos, freq = 500, n_delay = 5, r2_th = 0.6):
    total_weights = _cluster_weighting(xpos, ypos, n_delay, freq=freq)
    fix = _fix_correction(total_weights, freq, xpos, ypos)
    total_weights_corr = np.zeros(len(total_weights))
    for i in range(0, len(fix['end']) - 1):
        total_weights_corr[fix['end'][i]:fix['start'][i + 1]] = 1

    sac_info = _win_approx(total_weights_corr, xpos, ypos, freq=freq, offset_time=0.3)

    r2_max = np.maximum(np.array(sac_info['r2_x']), np.array(sac_info['r2_y']))
    indices = np.where(r2_max > r2_th)[0]
    sac_filtered = {key: [value[i] for i in indices] for key, value in sac_info.items()}

    return sac_filtered

def bool2bounds(b):
    b = np.array(np.array(b, dtype = bool), dtype=int)
    b = np.pad(b, (1, 1), 'constant', constant_values=(0, 0))
    D = np.diff(b)
    on  = np.array(np.where(D == 1)[0], dtype=int)
    off = np.array(np.where(D == -1)[0] -1, dtype=int)
    return on, off

def _fix_correction(final_weights, freq, xpos, ypos):
    timestamp = 1000 * np.arange(0, len(xpos)) / freq
    onoffsetThresh = 1
    maxMergeDist = 20.0
    maxMergeTime = 30.0
    minFixDur = 40.0

    fixbool = ~(final_weights.astype(bool))
    fixstart, fixend = bool2bounds(fixbool)

    for p in range(len(fixstart)):
        xFix = xpos[fixstart[p]:fixend[p]+1]
        yFix = ypos[fixstart[p]:fixend[p]+1]
        xmedThis = np.nanmedian(xFix)
        ymedThis = np.nanmedian(yFix)

        MAD = np.nanmedian(np.hypot(xFix-xmedThis, yFix-ymedThis))
        thresh = MAD*onoffsetThresh

        i = fixstart[p]
        if i>0:  # don't walk when fixation starting at start of data
            while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
                i = i+1
            fixstart[p] = i

        # and now fixation end.
        i = fixend[p]
        if i<len(xpos): # don't walk when fixation ending at end of data
            while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
                i = i-1
            fixend[p] = i

    ### get start time, end time,
    starttime = timestamp[fixstart]
    endtime = timestamp[fixend]

    ### loop over all fixation candidates in trial, see if should be merged
    for p in range(1,len(starttime))[::-1]:
        # get median coordinates of fixation
        xmedThis = np.median(xpos[fixstart[p]:fixend[p]+1])
        ymedThis = np.median(ypos[fixstart[p]:fixend[p]+1])
        xmedPrev = np.median(xpos[fixstart[p-1]:fixend[p-1]+1])
        ymedPrev = np.median(ypos[fixstart[p-1]:fixend[p-1]+1])

        if starttime[p]-endtime[p-1] < maxMergeTime and \
            np.hypot(xmedThis-xmedPrev,ymedThis-ymedPrev) < maxMergeDist:
            # merge
            fixend[p-1] = fixend[p]
            endtime[p-1]= endtime[p]
            # delete merged fixation
            fixstart = np.delete(fixstart, p)
            fixend = np.delete(fixend, p)
            starttime = np.delete(starttime, p)
            endtime = np.delete(endtime, p)

    # 1. determine what the duration of each end sample was
    nextSamp = np.min(np.vstack([fixend+1,np.zeros(len(fixend),dtype=int)+len(timestamp)-1]),axis=0) # make sure we don't run off the end of the data
    extratime = timestamp[nextSamp]-timestamp[fixend]

    if not len(fixend)==0 and fixend[-1]==len(timestamp): # first check if there are fixations in the first place, or we'll index into non-existing data
        extratime[-1] = np.diff(timestamp[-3:-1])

    # now add the duration of the sample to each fixation end time, so that they are
    # correct
    endtime = endtime+extratime

    ### calculate fixation duration
    fixdur = endtime-starttime

    ### check if any fixations are too short
    qTooShort = np.argwhere(fixdur<minFixDur)
    if len(qTooShort) > 0:
        fixstart = np.delete(fixstart, qTooShort)
        fixend = np.delete(fixend, qTooShort)
        starttime = np.delete(starttime, qTooShort)
        endtime = np.delete(endtime, qTooShort)
        fixdur = np.delete(fixdur, qTooShort)

    ### process fixations, get other info about them
    xmedian = np.zeros(fixstart.shape) # vector for median
    ymedian = np.zeros(fixstart.shape)  # vector for median
    for a in range(len(fixstart)):
        idxs = range(fixstart[a],fixend[a]+1)
        # get data during fixation
        xposf = xpos[idxs]
        yposf = ypos[idxs]
        xmedian[a] = np.median(xposf)
        ymedian[a] = np.median(yposf)

    # store all the results in a dictionary
    fix = {}
    fix['start'] = fixstart
    fix['end'] = fixend
    fix['startT'] = np.array(starttime)
    fix['endT'] = np.array(endtime)
    fix['dur'] = np.array(fixdur)
    fix['xpos'] = xmedian
    fix['ypos'] = ymedian
    return fix




