# from src.models.audio_flamingo import AudioFlamingo2
from src.models.baichuan import BaichuanAudio
from src.models.cascaded_model import CascadedModel
from src.models.desta2 import DeSTA2
from src.models.freeze_omni import FreezeOmni
from src.models.gemini import GeminiAudio
from src.models.glm4voice import Glm4Voice
from src.models.gpt4o import GPT4oAudio
from src.models.qwen2 import Qwen2Audio
from src.models.qwen25_omni import Qwen25Omni
from src.models.salmonn import SALMONN
from src.models.phi4 import Phi4Multimodal
from src.models.ultravox import Ultralvox

models_map = {
    'cascaded_model': CascadedModel,
    # API
    'gemini': GeminiAudio, # gemini api
    'gpt': GPT4oAudio, # gpt-4o-audio api

    # SLLM
    'qwen2-audio': Qwen2Audio,
    'baichuan_audio': BaichuanAudio,
    'qwen25-omni': Qwen25Omni, # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    'salmonn': SALMONN, # https://github.com/bytedance/SALMONN
    'glm': Glm4Voice, # https://huggingface.co/THUDM/glm-4-voice-9b
    'phi4': Phi4Multimodal, # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    'freeze_omni': FreezeOmni, # https://huggingface.co/VITA-MLLM/Freeze-Omni
    'desta2': DeSTA2, # https://huggingface.co/DeSTA-ntu/DeSTA2-8B-beta
    # 'audio-flamingo': AudioFlamingo2, will cause init very slow 
    'ultravox': Ultralvox, # https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b
}