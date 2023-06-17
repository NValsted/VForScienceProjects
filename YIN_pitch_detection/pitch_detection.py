import json

import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from matplotlib import pyplot as plt

from helper import PitchData, import_audio, plot_pitches


def ACF(f, W, t, lag):    
    return np.sum(
        f[t : t + W] *
        f[lag + t : lag + t + W]
    )


def detect_pitch_ACF(f, W, t, sample_rate, bounds):
    ACF_vals = [ACF(f, W, t, i) for i in range(*bounds)]
    sample = np.argmax(ACF_vals) + bounds[0]
    return sample_rate / sample


def DF(f, W, t, lag):
    return ACF(f, W, t, 0)\
        + ACF(f, W, t + lag, 0)\
        - (2 * ACF(f, W, t, lag))


def detect_pitch_DF(f, W, t, sample_rate, bounds):
    DF_vals = [DF(f, W, t, i) for i in range(1, sample_rate*3)]
    lower_bound = max(bounds[0], 0)
    upper_bound = min(bounds[1], len(DF_vals))
    sample = np.argmin(
        DF_vals[lower_bound:upper_bound]
    ) + lower_bound
    return sample_rate / sample


def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1
    return DF(f, W, t, lag)\
        / np.sum([DF(f, W, t, j + 1) for j in range(lag)]) * lag


def detect_pitch_CMNDF(f, W, t, sample_rate, bounds):
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(1, sample_rate*3)]
    lower_bound = max(bounds[0], 0)
    upper_bound = min(bounds[1], len(CMNDF_vals))
    sample = np.argmin(
        CMNDF_vals[lower_bound:upper_bound]
    ) + lower_bound
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


def augmented_detect_pitch_CMNDF(f, W, t, sample_rate, bounds, thresh=0.1):  # Also uses memoization
    CMNDF_vals = memo_CMNDF(f, W, t, bounds[-1])[bounds[0]:]
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
    corr = np.correlate(f[t : t + W], f[t + lag : t + lag + W], mode = 'valid')
    return corr[0]


def DF_lesm(f, W, lag):
    return ACF_lesm(f, W, 0, 0) + ACF_lesm(f, W, lag, 0) - (2 * ACF_lesm(f, W, 0, lag))


# Optimized Algorithm without Parabolic Interpolation or Best Local Estimate
def detect_pitch_lesm(f, W, sample_rate, bounds, thresh=0.1):
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


# Optimized Algorithm with Parabolic Interpolation
def detect_pitch_interpolated_lesm(f, W, sample_rate, bounds, thresh=0.1):
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
        sample = np.argmin(vals[bounds[0]:]) + bounds[0]
    
    # Parabolic interpolation
    if 1 < sample < len(vals) - 1:
        s0, s1, s2 = vals[sample-1], vals[sample], vals[sample+1]
        correction = 0.5 * (s2 - s0) / (2 * s1 - s2 - s0)
        sample += correction

    return sample_rate / sample


def process_audio(data, sample_rate, name="" ,fmin=100, fmax=2000, windows_size_ms=2.5, method='lesm'):
    # Resulting values
    windows_size = int(windows_size_ms/1000 * sample_rate)
    lagMin = int(1/fmax * sample_rate)
    lagMax = int(1/fmin * sample_rate)
    bounds = [ lagMin , lagMax ]

    pitches = []
    
    if method == 'ACF':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            pitches.append(detect_pitch_ACF(f=data, W=windows_size, t=i*windows_size, sample_rate=sample_rate, bounds=bounds))
    elif method == 'DF':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            pitches.append(detect_pitch_DF(f=data, W=windows_size, t=i*windows_size, sample_rate=sample_rate, bounds=bounds))
    elif method == 'CMNDF':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            pitches.append(detect_pitch_CMNDF(f=data, W=windows_size, t=i*windows_size, sample_rate=sample_rate, bounds=bounds))
    elif method == 'CMNDF-memo':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            pitches.append(augmented_detect_pitch_CMNDF(f=data, W=windows_size, t=i*windows_size, sample_rate=sample_rate, bounds=bounds))
    elif method == 'lesm':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            t = i*windows_size
            pitches.append(detect_pitch_lesm(f=data[t : t + windows_size + lagMax], W=windows_size, sample_rate=sample_rate, bounds=bounds))
    elif method == 'lesm-i':
        for i in tqdm(range(data.shape[0] // (windows_size+3))):
            t = i*windows_size
            pitches.append(detect_pitch_interpolated_lesm(f=data[t : t + windows_size + lagMax], W=windows_size, sample_rate=sample_rate, bounds=bounds))
    else:
        raise ValueError(f'Invalid method: {method}')

    return PitchData(name=(name if name is not None else ""), sampleRate=sample_rate, srcLen=data.shape[0], pitches=np.array(pitches))


def f(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return np.sin(x * np.pi * 2 * f_0) * envelope(x)


def synthesized_signal_main():
    sample_rate = 500
    start = 0
    end = 5
    num_samples = int(sample_rate * (end - start) + 1)
    window_size = 200
    bounds = [20, num_samples // 2]
    x = np.linspace(start, end, num_samples)

    for label, detection_function in [
        ("ACF", detect_pitch_ACF),
        ("DF", detect_pitch_DF),
        ("CMNDF", detect_pitch_CMNDF),
        ("CMNDF and thresholding", augmented_detect_pitch_CMNDF)
    ]:
        print(
            f"Detected pitch with {label}: "
            f"{detection_function(f(x), window_size, 1, sample_rate, bounds)}"
        )


def singer_main():
    sample_rate, data = wavfile.read("YIN_pitch_detection/singer.wav")
    data = data.astype(np.float64)
    window_size = int(5 / 2000 * 44100)
    bounds = [20, 2000]

    pitches = []
    for i in tqdm(range(data.shape[0] // (window_size + 3))):
        pitches.append(
            augmented_detect_pitch_CMNDF(
                data,
                window_size,
                i * window_size,
                sample_rate,
                bounds
            )
        )

    with open("pitch_vals.json", "w") as file:
        json.dump(pitches, file)

    plt.plot(pitches)
    plt.ylim(300, 600)
    plt.show()


def lesm_main():
    audio_file = "singer.wav"
    methods = ['lesm']
    fs, audio_data = import_audio(audio_file)
    singer_pitches = []
    for m in methods:
        print('Processing audio file: ' + audio_file + ' with method: ' + m + '...')
        singer_pitches.append(process_audio(data = audio_data, sample_rate = fs, name=audio_file+' '+m ,fmin=20, fmax=2000, windows_size_ms=2.5, method=m))
    plot_pitches(fmin=300, fmax=600, dataObjects=singer_pitches)


if __name__ == '__main__':
    synthesized_signal_main()
    singer_main()
    lesm_main()
