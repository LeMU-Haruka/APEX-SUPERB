from importlib import import_module


class LazyModel:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name

    def __call__(self, *args, **kwargs):
        module = import_module(self.module_name)
        model_cls = getattr(module, self.class_name)
        return model_cls(*args, **kwargs)


models_map = {
    'whisper': LazyModel('src.models.whisper', 'Whisper'),
    'cascaded-llama3': LazyModel('src.models.cascaded_llama3', 'CascadedLlama3'),
    'cascaded-qwen2': LazyModel('src.models.cascaded_qwen2', 'CascadedQwen2'),
    'cascaded-qwen25': LazyModel('src.models.cascaded_qwen2', 'CascadedQwen2'),

    # API
    'gemini': LazyModel('src.models.gemini', 'GeminiAudio'), # gemini api
    'gpt': LazyModel('src.models.gpt4o', 'GPT4oAudio'), # gpt-4o-audio api

    # SLLM
    'qwen2-audio': LazyModel('src.models.qwen2', 'Qwen2Audio'), # https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
    'baichuan-audio': LazyModel('src.models.baichuan', 'BaichuanAudio'), # https://huggingface.co/baichuan-inc/Baichuan-Audio-Instruct
    'qwen25-omni': LazyModel('src.models.qwen25_omni', 'Qwen25Omni'), # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    'salmonn': LazyModel('src.models.salmonn', 'SALMONN'), # https://github.com/bytedance/SALMONN
    'glm': LazyModel('src.models.glm4voice', 'Glm4Voice'), # https://huggingface.co/THUDM/glm-4-voice-9b
    'phi4': LazyModel('src.models.phi4', 'Phi4Multimodal'), # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    'freeze-omni': LazyModel('src.models.freeze_omni', 'FreezeOmni'), # https://huggingface.co/VITA-MLLM/Freeze-Omni
    'desta2': LazyModel('src.models.desta2', 'DeSTA2'), # https://huggingface.co/DeSTA-ntu/DeSTA2-8B-beta
    'audio-flamingo': LazyModel('src.models.audio_flamingo', 'AudioFlamingo2'), # will cause init very slow
    'ultravox': LazyModel('src.models.ultravox', 'Ultralvox'), # https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b
    'kimi-audio': LazyModel('src.models.kimi_audio', 'Kimi'), # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
    'aero-audio': LazyModel('src.models.aero_audio', 'AeroAudio'), # https://huggingface.co/lmms-lab/Aero-1-Audio
}
