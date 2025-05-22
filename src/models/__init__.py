# from src.models.audio_flamingo import AudioFlamingo2
from src.models.aero_audio import AeroAudio
from src.models.baichuan import BaichuanAudio
from src.models.cascaded_llama3 import CascadedLlama3
from src.models.cascaded_qwen2 import CascadedQwen2
from src.models.desta2 import DeSTA2
from src.models.freeze_omni import FreezeOmni
from src.models.gemini import GeminiAudio
from src.models.glm4voice import Glm4Voice
from src.models.gpt4o import GPT4oAudio
from src.models.kimi_audio import Kimi
from src.models.qwen2 import Qwen2Audio
from src.models.qwen25_omni import Qwen25Omni
from src.models.salmonn import SALMONN
from src.models.phi4 import Phi4Multimodal
from src.models.ultravox import Ultralvox
from src.models.whisper import Whisper


models_map = {
    'whisper': Whisper,
    'cascaded-llama3': CascadedLlama3,
    'cascaded-qwen2': CascadedQwen2,
    # API
    'gemini': GeminiAudio, # gemini api
    'gpt': GPT4oAudio, # gpt-4o-audio api

    # SLLM
    'qwen2-audio': Qwen2Audio, # https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
    'baichuan-audio': BaichuanAudio, # https://huggingface.co/baichuan-inc/Baichuan-Audio-Instruct
    'qwen25-omni': Qwen25Omni, # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    'salmonn': SALMONN, # https://github.com/bytedance/SALMONN
    'glm': Glm4Voice, # https://huggingface.co/THUDM/glm-4-voice-9b
    'phi4': Phi4Multimodal, # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    'freeze-omni': FreezeOmni, # https://huggingface.co/VITA-MLLM/Freeze-Omni
    'desta2': DeSTA2, # https://huggingface.co/DeSTA-ntu/DeSTA2-8B-beta
    # 'audio-flamingo': AudioFlamingo2, will cause init very slow 
    'ultravox': Ultralvox, # https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b
    'kimi-audio': Kimi, # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
    'aero-audio': AeroAudio, # https://huggingface.co/lmms-lab/Aero-1-Audio
}