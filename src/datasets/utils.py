import soundfile as sf



def load_audio_file(file):
    waveform, sample_rate = sf.read(file)
    return waveform, sample_rate