import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

from src.models.base_model import BaseModel


class Whisper(BaseModel):
    def __init__(self, llm_path='/userhome/models/whisper-large-v3'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.asr_pipe = self.init_asr_pipeline(llm_path)


    def init_asr_pipeline(self, whisper_path):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(whisper_path)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
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

            
    def chat_mode(
        self,
        audio,
        max_new_tokens=2048,
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
            max_new_tokens=2048,
    ):
        asr_text = self.asr(audio, sr)
        return asr_text


    def text_mode(self, prompt, text, max_new_tokens=2048):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
