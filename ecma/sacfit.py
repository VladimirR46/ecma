"""
@author: Antipov Vladimir, Badarin Artem
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def f(t):
    return np.where(t >= 0, t + 0.25 * np.exp(-2 * t), 0.25 * np.exp(2 * t))

def saccadic_waveform(t, eta, c, tau):
    waveform = c*f(eta*t/c) - c*f(eta*(t-tau)/c)
    return waveform

def waveform_xy(t, dur, c, amp, t0=None):
    eta = c * np.log(0.5 / 0.01 * (np.exp(amp / c) + 1)) / dur
    tau = amp / eta

    if t0 is None:
        t0 = -tau/2

    waveform = saccadic_waveform(t - t0, eta, c, tau)
    return waveform

def saccade_model_xy(t, dur, c, amp, alpha, k, t0=None, s0x=0, s0y=0):
    eta = c * np.log(0.5 / 0.01 * (np.exp(amp / c) + 1)) / dur
    tau = amp / eta

    if t0 is None:
        t0 = -tau/2

    waveform = saccadic_waveform(t - t0, eta, c, tau)
    sacc_x = waveform * np.cos(alpha) + s0x
    sacc_y = k*waveform * np.sin(alpha) + s0y
    return np.concatenate((sacc_x, sacc_y))

def saccade_fit_xy(saccade_x, saccade_y, sampling_rate):
    duration = len(saccade_x) / sampling_rate
    t = np.linspace(-duration / 2, duration / 2, len(saccade_x))

    data_xy = np.concatenate((saccade_x, saccade_y))

    s0x_min, s0x_max = saccade_x.min(), saccade_x.max()
    s0y_min, s0y_max = saccade_y.min(), saccade_y.max()

    t0_min, t0_max = t[0], t[-1]
    c_min, c_max = 2, 1000
    amp_min, amp_max = 0.1, data_xy.max() - data_xy.min()  # 1, 35000#
    dur_min, dur_max = 0.01, 0.150
    alpha_min, alpha_max = -np.pi, np.pi
    k_min, k_max = 0.9, 1.1

    bounds = ([dur_min, c_min, amp_min, alpha_min, k_min, t0_min, s0x_min, s0y_min],
              [dur_max, c_max, amp_max, alpha_max, k_max, t0_max, s0x_max, s0y_max])

    try:
        popt, pcov = curve_fit(saccade_model_xy, t, data_xy, method='trf', bounds=bounds)
    except:
        #print("except curve_fit")
        return None

    optimized_data = saccade_model_xy(t, *popt)
    sac_x, sac_y = optimized_data[:len(t)], optimized_data[len(t):]
    waveform = waveform_xy(t, popt[0], popt[1], popt[2], popt[5])

    vel = np.diff(waveform) * sampling_rate
    vel = np.abs(np.insert(vel, 0, 0))

    threshold = np.max(vel) * 0.01
    left_index = np.argmax(vel > threshold)
    right_index = len(vel) - 1 - np.argmax(vel[::-1] > threshold)

    info = {
        'sac': waveform,
        'sac_x': sac_x,
        'sac_y': sac_y,
        'dur': t[right_index]-t[left_index],
        'amp': np.abs(waveform[0]-waveform[-1]),
        'vel': vel,
        'r2': np.mean(r2_score(saccade_x, sac_x)+r2_score(saccade_y, sac_y)),
        'begin_peak': left_index,
        'end_peak': right_index,
        'peak': round((left_index+right_index)/2),
        'dist': xy_to_dist(sac_x[left_index], sac_y[left_index], sac_x[right_index], sac_y[right_index])
    }

    return info

def xy_to_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)



