import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, AutoModelForCausalLM, AutoTokenizer

from src.models.base_model import BaseModel


class CascadedModel(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct', whisper_path='/userhome/models/whisper-large-v3'):
        self.asr_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.llm_device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.asr_pipe = self.init_asr_model(whisper_path)

        # init text LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 将 pad_token 设置为 eos_token
        self.model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.model.to(self.llm_device)
        self.model.eval()


    def init_asr_model(self, whisper_path):  
        processor = AutoProcessor.from_pretrained(whisper_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        model.to(self.asr_device)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=self.torch_dtype,
            device=self.asr_device,
        )
        return pipe

    def asr(self, audio, sr):
        assert sr == 16000
        result = self.asr_pipe(audio, generate_kwargs={"language": "english", "task": "transcribe"})
        text = result['text']
        return text
    

    def prompt_fomat(self, prompt, question):
        # prompt_template = """
        # <s>[INST] <<SYS>>
        # You are a helpful assistant.
        # <</SYS>>

        # <PROMPT> 
        # The question is: <QUESTION>
        #  [/INST]
        # """
        prompt_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            
        Cutting Knowledge Date: December 2023
        Today Date: 26 Jul 2024
        You are a professional assistant for evaluating large model generations. Please carefully analyze and respond based on the given prompt.<|eot_id|><|start_header_id|>user<|end_header_id|>
        [PROMPT] 
        The question is: <QUESTION>
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        prompt_template = prompt_template.replace('<QUESTION>', question)
        prompt_template = prompt_template.replace('<PROMPT>', prompt)
        return prompt_template 

    def generate(self, prompt):
        # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**input_ids, max_new_tokens=200, cache_implementation="static")
        # response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # return response
        model_input = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.llm_device)
        with torch.no_grad():
            outputs = self.model.generate(**model_input, max_new_tokens=300, temperature=0.1, top_p=0.75)
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(decoded_outputs)
        return decoded_outputs[0]
        
            
    def chat_mode(
        self,
        audio,
        max_new_tokens=2048,
    ):
        print ('cascaded model do not have chat mode')
        pass

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=2048,
    ):
        asr_text = self.asr(audio, sr)
        # prompt = prompt + ' The question is: ' + asr_text + '\nAnswer: '
        prompt = self.prompt_fomat(prompt, asr_text)
        response = self.generate(prompt)
        return response


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


# debug