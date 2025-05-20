import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, AutoModelForCausalLM, AutoTokenizer

from src.models.base_model import BaseModel


class CascadedQwen2(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-7B-Instruct', whisper_path='openai/whisper-large-v3'):
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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "### Task Instruction\n" + prompt + "\n\n### Question\n" + question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_device)
        return model_inputs 

    def generate(self, model_inputs):
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,   # <|eot_id|>
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response.strip())
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
        # prompt = prompt + ' The question is: ' + asr_text + '\nAnswer: '
        prompt = self.prompt_fomat(prompt, asr_text)
        response = self.generate(prompt)
        return response


    def text_mode(self, prompt, text, max_new_tokens=1024):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
