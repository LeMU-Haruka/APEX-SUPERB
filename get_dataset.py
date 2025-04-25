import json

with open("data/dataset/baichuan_audio_asr_commonvoice.json", 'r') as f:
    commonvoice = json.load(f)
    for item in commonvoice:
        with open("data/dataset/commonvoice_audios.txt", 'a') as f2:
            f2.write(item["target"]+"\n")
            
with open("data/dataset/baichuan_audio_asr_librispeech.json", 'r') as f:
    librispeech = json.load(f)
    for item in librispeech:
        with open("data/dataset/librispeech_audios.txt", 'a') as f2:
            f2.write(item["target"]+"\n")
        