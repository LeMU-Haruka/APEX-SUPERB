import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, AutoModelForCausalLM, AutoTokenizer

from src.models.base_model import BaseModel
from src.models.ssalmonn_modules.models.modeling_whisper import WhisperForConditionalGeneration


class CascadedLlama3(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct', whisper_path='/userhome/models/whisper-large-v3'):
        self.asr_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.llm_device = torch.device("cuda:1")
        print(f"ASR device: {self.asr_device}, LLM device: {self.llm_device}")
        self.torch_dtype = torch.float16 if self.asr_device.type == "cuda" else torch.float32
        self.asr_pipe = self.init_asr_pipeline(whisper_path)

        # init text LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 将 pad_token 设置为 eos_token
        self.model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.model.to(self.llm_device)
        self.model.eval()


    def init_asr_pipeline(self, whisper_path):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.asr_device)

        processor = AutoProcessor.from_pretrained(whisper_path)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.asr_device,
            chunk_length_s=30,
            stride_length_s=6,  
            return_timestamps=False,
        )
        return pipe

    def asr(self, audio, sr):
        assert sr == 16000
        result = self.asr_pipe(audio)
        transcription = result["text"]
        return transcription

    def prompt_fomat(self, prompt, question):
        messages = [
            {"role": "system",
            "content": "You are a concise, knowledgeable assistant."},
            {"role": "user",
            "content": "### Task Instruction\n" + prompt + "\n\n### Question\n" + question},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        return prompt_ids 

    def generate(self, prompt):
        out_ids = self.model.generate(
            prompt,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(out_ids[0][prompt.shape[-1]:], skip_special_tokens=True)
        return response
                
            
    def chat_mode(
        self,
        audio,
        max_new_tokens=1024,
    ):
        asr_text = self.asr(audio)
        prompt = self.prompt_fomat('', asr_text)
        response = self.generate(prompt)
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        asr_text = self.asr(audio, sr)
        prompt = self.prompt_fomat(prompt, asr_text)
        response = self.generate(prompt)
        return response