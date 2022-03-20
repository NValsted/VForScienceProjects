import typer

app = typer.Typer()

SAMPLE_RATE = 44100
BUFFER_SIZE = 128
BPM = 128
DURATION = 0.2


@app.command()
def mimic(
    audio_path: str = typer.Option(
        ..., help="Path to an audio file that should be mimicked by a piano"
    ),
    data_dir: str = typer.Option(..., help="Path to the data directory"),
):
    from collections import defaultdict
    from glob import glob
    from re import compile

    import numpy as np
    from scipy.fft import rfft
    from scipy.io import wavfile

    from genetic import fit_combination
    from utils import chunk_iterator

    note_regex = compile(r"note\d+\_")
    chunk_size = int(SAMPLE_RATE * DURATION)

    component_files = glob(f"{data_dir}/fft/*.npy")
    components = defaultdict(list)
    for component_file in component_files:
        note = note_regex.search(component_file).group(0)
        components[note].append(np.load(component_file))

    sample_rate, target = wavfile.read(audio_path)
    if int(sample_rate) != int(SAMPLE_RATE):
        raise ValueError(
            f"Sample rate of {audio_path} is {sample_rate}, but expected {SAMPLE_RATE}"
        )

    for chunk in chunk_iterator(target, chunk_size):
        if chunk.shape[0] != chunk_size:
            continue
        transformed = np.abs(rfft(chunk))
        best_score, best_solution = fit_combination(transformed, components)


@app.command()
def prepare_components(
    data_dir: str = typer.Option(..., help="Path to the data directory"),
    vst_path: str = typer.Option(..., help="Path to the VST plugin"),
):
    import numpy as np
    from dawdreamer import RenderEngine
    from mido import Message, MidiFile, MidiTrack
    from scipy.fft import rfft
    from scipy.io import wavfile
    from tqdm import tqdm

    # Generate MIDI input
    midi_files = []

    for note in tqdm(range(21, 109)):
        for velocity in range(0, 128):
            track = MidiTrack()
            track.append(Message("note_on", note=note, velocity=velocity, time=0))
            track.append(Message("note_off", note=note, velocity=velocity, time=100))

            file = MidiFile(type=0)
            file.tracks.append(track)
            save_path = f"{data_dir}/midi/note{note}_velocity{velocity}.mid"

            file.save(save_path)
            midi_files.append(save_path)

    # Synthesize piano sounds
    engine = RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    engine.set_bpm(BPM)
    synth = engine.make_plugin_processor("synth", vst_path)
    engine.load_graph([(synth, [])])

    for file in tqdm(midi_files):
        synth.load_midi(file)
        engine.render(DURATION)
        audio = engine.get_audio().transpose()[:, 0]

        wavfile.write(
            file.replace("/midi/", "/audio/").replace(".mid", ".wav"),
            SAMPLE_RATE,
            audio,
        )

        # Process audio
        transformed = np.abs(rfft(audio))
        with open(file.replace("/midi/", "/fft/").replace(".mid", ".npy"), "wb") as f:
            np.save(f, transformed)


if __name__ == "__main__":
    app()
