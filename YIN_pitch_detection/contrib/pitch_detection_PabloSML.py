# Contributions by https://github.com/PabloSML
#
# Comments from original PR (https://github.com/NValsted/VForScienceProjects/pull/1):
#
#    - Add helper.py with PitchData class, import_audio and plot_pitches functions.
#    - Add "lesm" set of optimized functions for real time pitch detection.
#    - Add process_audio function for easier use of all detection functions.
#    - Add lesm_main for testing of new features.
#
# This was done as part of a voice processing course assignment by my partner F. Ledesma
# and me, P. Smolkin. Hence the name "LESM" for all the changes, which doesn't have to
# stay that way necessarily.
#
# The main changes on performance were avoiding unnecessary repetitions of the same
# calculations, as well as not running across the data once for calculating values for
# tau and then again for selecting the estimate.
#
# We also added some plotting functions, and the detect pitch functions do not recieve
# the full data each time, rather just the required window, which would allow real time
# processing with a minimum lag of w_size + maxlag (12.5ms with our parameters), as
# stated by the original paper.

import numpy as np
from tqdm import tqdm

from helper import PitchData, import_audio, plot_pitches


def ACF(f, W, t, lag):
    return np.sum(f[t : t + W] * f[lag + t : lag + t + W])


def detect_pitch_ACF(f, W, t, sample_rate, bounds):
    ACF_vals = [ACF(f, W, t, i) for i in range(*bounds)]
    sample = np.argmax(ACF_vals) + bounds[0]
    return sample_rate / sample


def DF(f, W, t, lag):
    return ACF(f, W, t, 0) + ACF(f, W, t + lag, 0) - (2 * ACF(f, W, t, lag))


def detect_pitch_DF(f, W, t, sample_rate, bounds):
    DF_vals = [DF(f, W, t, i) for i in range(1, sample_rate * 3)]
    lower_bound = max(bounds[0], 0)
    upper_bound = min(bounds[1], len(DF_vals))
    sample = np.argmin(DF_vals[lower_bound:upper_bound]) + lower_bound
    return sample_rate / sample


def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1
    return DF(f, W, t, lag) / np.sum([DF(f, W, t, j + 1) for j in range(lag)]) * lag


def detect_pitch_CMNDF(f, W, t, sample_rate, bounds):
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(1, sample_rate * 3)]
    lower_bound = max(bounds[0], 0)
    upper_bound = min(bounds[1], len(CMNDF_vals))
    sample = np.argmin(CMNDF_vals[lower_bound:upper_bound]) + lower_bound
    return sample_rate / sample


def memo_CMNDF(f, W, t, lag_max):
    running_sum = 0
    vals = []
    for lag in range(0, lag_max):
        if lag == 0:
            vals.append(1)
            running_sum += 0
        else:
            running_sum += DF(f, W, t, lag)
            vals.append(DF(f, W, t, lag) / running_sum * lag)
    return vals


def augmented_detect_pitch_CMNDF(
    f, W, t, sample_rate, bounds, thresh=0.1
):  # Also uses memoization
    CMNDF_vals = memo_CMNDF(f, W, t, bounds[-1])[bounds[0] :]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < thresh:
            sample = i + bounds[0]
            break
    if sample is None:
        sample = np.argmin(CMNDF_vals) + bounds[0]
    return sample_rate / (sample + 1)


# Optimized functions for Real Time - LESM (Ledesma-Smolkin / Less Math)
# Note detect funtions only need recieve the data from t to t + W + lagMax


def ACF_lesm(f, W, t, lag):
    corr = np.correlate(f[t : t + W], f[t + lag : t + lag + W], mode="valid")
    return corr[0]


def DF_lesm(f, W, lag):
    return ACF_lesm(f, W, 0, 0) + ACF_lesm(f, W, lag, 0) - (2 * ACF_lesm(f, W, 0, lag))


def detect_pitch_lesm(f, W, sample_rate, bounds, thresh=0.1):
    """
    Optimized Algorithm without Parabolic Interpolation or Best Local Estimate
    """
    lag_max = bounds[1]
    running_sum = 0
    vals = [1]
    sample = None

    for lag in range(1, lag_max):
        # Difference Function
        dfResult = DF_lesm(f, W, lag)
        # Memoized Cumulative Mean Normalized Difference Function
        running_sum += dfResult
        val = dfResult / running_sum * lag
        vals.append(val)
        # Absolute Thresholding with short-stopping
        if lag >= bounds[0] and val < thresh:
            sample = lag
            break
    # No acceptable lag found, default to minimum error
    else:
        argmin = np.argmin(vals)
        sample = argmin if argmin > bounds[0] else bounds[0]

    return sample_rate / sample


def detect_pitch_interpolated_lesm(f, W, sample_rate, bounds, thresh=0.1):
    """
    Optimized Algorithm with Parabolic Interpolation
    """
    lag_max = bounds[1]
    running_sum = 0
    vals = [1]

    for lag in range(1, lag_max):
        # Difference Function
        dfResult = DF_lesm(f, W, lag)
        # Memoized Cumulative Mean Normalized Difference Function
        running_sum += dfResult
        val = dfResult / running_sum * lag
        vals.append(val)
        # Absolute Thresholding with short-stopping
        if lag >= bounds[0] and val < thresh:
            sample = lag
            break
    # No acceptable lag found, default to minimum error
    else:
        sample = np.argmin(vals[bounds[0] :]) + bounds[0]

    # Parabolic interpolation
    if 1 < sample < len(vals) - 1:
        s0, s1, s2 = vals[sample - 1], vals[sample], vals[sample + 1]
        correction = 0.5 * (s2 - s0) / (2 * s1 - s2 - s0)
        sample += correction

    return sample_rate / sample


def process_audio(
    data, sample_rate, name="", fmin=100, fmax=2000, windows_size_ms=2.5, method="lesm"
):
    # Resulting values
    windows_size = int(windows_size_ms / 1000 * sample_rate)
    lagMin = int(1 / fmax * sample_rate)
    lagMax = int(1 / fmin * sample_rate)
    bounds = [lagMin, lagMax]

    pitches = []

    if method == "ACF":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            pitches.append(
                detect_pitch_ACF(
                    f=data,
                    W=windows_size,
                    t=i * windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    elif method == "DF":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            pitches.append(
                detect_pitch_DF(
                    f=data,
                    W=windows_size,
                    t=i * windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    elif method == "CMNDF":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            pitches.append(
                detect_pitch_CMNDF(
                    f=data,
                    W=windows_size,
                    t=i * windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    elif method == "CMNDF-memo":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            pitches.append(
                augmented_detect_pitch_CMNDF(
                    f=data,
                    W=windows_size,
                    t=i * windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    elif method == "lesm":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            t = i * windows_size
            pitches.append(
                detect_pitch_lesm(
                    f=data[t : t + windows_size + lagMax],
                    W=windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    elif method == "lesm-i":
        for i in tqdm(range(data.shape[0] // (windows_size + 3))):
            t = i * windows_size
            pitches.append(
                detect_pitch_interpolated_lesm(
                    f=data[t : t + windows_size + lagMax],
                    W=windows_size,
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            )
    else:
        raise ValueError(f"Invalid method: {method}")

    return PitchData(
        name=(name if name is not None else ""),
        sampleRate=sample_rate,
        srcLen=data.shape[0],
        pitches=np.array(pitches),
    )


def f(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return np.sin(x * np.pi * 2 * f_0) * envelope(x)


def lesm_main():
    audio_file = "singer.wav"
    methods = ["lesm"]
    fs, audio_data = import_audio(audio_file)
    singer_pitches = []
    for m in methods:
        print("Processing audio file: " + audio_file + " with method: " + m + "...")
        singer_pitches.append(
            process_audio(
                data=audio_data,
                sample_rate=fs,
                name=audio_file + " " + m,
                fmin=20,
                fmax=2000,
                windows_size_ms=2.5,
                method=m,
            )
        )
    plot_pitches(fmin=300, fmax=600, dataObjects=singer_pitches)


if __name__ == "__main__":
    lesm_main()
