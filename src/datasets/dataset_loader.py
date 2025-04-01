from datasets import load_dataset, Audio

from src.datasets.librispeech import LibriSpeech


def load_librispeech_data():
    # change to librispeech root path
    dataset = LibriSpeech()
    return dataset




def load_asr_data():
    # data = load_dataset("openslr/librispeech_asr", 'clean', split="test", revision='refs/convert/parquet', trust_remote_code=True)
    data = load_librispeech_data()
    return data