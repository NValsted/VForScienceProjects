import numpy as np
from dataclasses import dataclass
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pathlib import Path


@dataclass(frozen=True)
class PitchData:
    """
    Data class for processed pitch data
    """

    name: str
    sampleRate: int
    srcLen: int
    pitches: np.ndarray


def import_audio(audio_file):
    """
    Import audio file and return sample rate and data
    """
    sample_rate, data = wavfile.read(
        str(Path(__file__).resolve().parent.parent / audio_file)
    )
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float64)
    return sample_rate, data


# Plotting functions

# fmt: off
# Numpy array of all note frequencies from C0 to B8
noteFrequencies = np.array([16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13])
# fmt: on

# Numpy array of intermediate frequencies between each note
halfFrequencies = np.array(
    [
        np.mean([noteFrequencies[i], noteFrequencies[i + 1]])
        for i in range(len(noteFrequencies) - 1)
    ]
)

# Dictionary of RGB colors for each note, using the same color for each octave
noteColors = {
    halfFrequencies[i]: plt.cm.tab20(i % 12) for i in range(len(halfFrequencies))
}

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def plot_note_intervals(fmin, fmax):
    """
    Function that plots the note intervals in color using same color for note octaves
    """
    idmin = np.abs(noteFrequencies - fmin).argmin()
    idmin = idmin - 2 if noteFrequencies[idmin] > fmin else idmin - 1
    idmax = np.abs(noteFrequencies - fmax).argmin()
    idmax = idmax if noteFrequencies[idmax] < fmax else idmax - 1

    for i in range(idmin, idmax):
        plt.axhspan(
            halfFrequencies[i],
            halfFrequencies[i + 1],
            color=noteColors[halfFrequencies[i]],
            alpha=0.2,
        )
        # Show note names in the middle of the interval
        if fmin < noteFrequencies[i + 1] < fmax:
            plt.text(
                0,
                noteFrequencies[i + 1],
                note_names[(i + 1) % 12] + str((i + 1) // 12),
                ha="right",
                va="center",
            )


def plot_pitches(fmin: int, fmax: int, dataObjects=None):
    if dataObjects is not None:
        plt.figure(figsize=(20, 8))
        plt.yscale("log")
        plot_note_intervals(fmin=fmin, fmax=fmax)
        for index, dataObject in enumerate(dataObjects):
            t = np.linspace(
                0,
                dataObject.srcLen // dataObject.sampleRate,
                num=len(dataObject.pitches),
            )
            plt.scatter(
                t,
                dataObject.pitches,
                color=plt.cm.Set1(index % 9),
                label=dataObject.name,
                s=10,
            )
        plt.legend()
        plt.ylim(fmin, fmax)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()
