import numpy as np
from _cffi_backend import _CDataBase
from sounddevice import CallbackFlags, OutputStream, sleep


class Synthesizer:
    SAMPLE_RATE: int = 44_100
    PHI: float = 0.0
    VOLUME: float = 0.05
    WAVETABLE_SIZE: int = 256
    FREQUENCY: float = 440.0

    TABLE_SIZE_OVER_SAMPLE_RATE: float = WAVETABLE_SIZE / SAMPLE_RATE
    TWO_PI_OVER_SAMPLE_RATE: float = 2 * np.pi / SAMPLE_RATE
    SIN_WAVETABLE = np.sin(2 * np.pi * np.arange(WAVETABLE_SIZE) / WAVETABLE_SIZE)

    def get_table_delta(self, frequency: float) -> float:
        return frequency * self.TABLE_SIZE_OVER_SAMPLE_RATE

    def get_angle_delta(self, frequency: float) -> float:
        return frequency * self.TWO_PI_OVER_SAMPLE_RATE

    def process_buffer(
        self,
        audio_buffer: np.ndarray,
        num_samples: int,
        time: _CDataBase,
        status: CallbackFlags,
    ) -> None:
        for i in range(num_samples):
            # audio_buffer[i, :] = self.SIN_WAVETABLE[int(self.PHI)] * self.VOLUME
            # self.PHI += self.get_table_delta(self.FREQUENCY)
            # self.PHI %= self.WAVETABLE_SIZE
            audio_buffer[i, :] = np.sin(self.PHI) * self.VOLUME
            self.PHI += self.get_angle_delta(self.FREQUENCY)

    def open_stream(self):
        with OutputStream(
            samplerate=self.SAMPLE_RATE, channels=2, callback=self.process_buffer
        ):
            sleep(int(1 * 1000))


if __name__ == "__main__":
    synth = Synthesizer()
    synth.open_stream()
