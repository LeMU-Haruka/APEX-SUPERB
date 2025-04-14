from src.models.baichuan import BaichuanAudio
from src.models.cascaded_model import CascadedModel
from src.models.freeze_omni import FreezeOmni
from src.models.gemini import GeminiAudio
from src.models.glm4voice import Glm4Voice
from src.models.mini_omni2 import MiniOmni2
from src.models.qwen2 import Qwen2Audio
from src.models.qwen25_omni import Qwen25Omni
from src.models.salmonn import SALMONN
from src.models.phi4 import Phi4Multimodal

models_map = {
    'cascaded_model': CascadedModel,
    # API
    'gemini': GeminiAudio, # gemini api

    # SLLM
    'qwen2-audio': Qwen2Audio,
    'baichuan_audio': BaichuanAudio,
    'mini-omni2': MiniOmni2,
    'qwen25-omni': Qwen25Omni, # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    'salmonn': SALMONN, # https://github.com/bytedance/SALMONN
    'glm': Glm4Voice, # https://huggingface.co/THUDM/glm-4-voice-9b
    'phi4': Phi4Multimodal, # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    'freeze_omni': FreezeOmni, # https://huggingface.co/VITA-MLLM/Freeze-Omni
}