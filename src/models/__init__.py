from src.models.baichuan import Baichuan
from src.models.mini_omni2 import MiniOmni2
from src.models.qwen2 import Qwen2Audio

models_map = {
    'qwen2-audio': Qwen2Audio,
    'baichuan_audio': Baichuan,
    'mini-omni2': MiniOmni2,
}