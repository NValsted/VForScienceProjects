import json

import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from matplotlib import pyplot as plt


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
    sample_rate, data = wavfile.read("singer.wav")
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


if __name__ == '__main__':
    synthesized_signal_main()
    singer_main()
