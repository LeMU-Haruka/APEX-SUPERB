from src.models.baichuan import Baichuan
from src.models.mini_omni2 import MiniOmni2
from src.models.qwen2 import Qwen2Audio
from src.models.qwen25_omni import Qwen25Omni
from src.models.salmonn import SALMONN

models_map = {
    'qwen2-audio': Qwen2Audio,
    'baichuan_audio': Baichuan,
    'mini-omni2': MiniOmni2,
    'qwen25-omni': Qwen25Omni, # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    'salmonn': SALMONN, # https://github.com/bytedance/SALMONN
}