import torch
from src.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class Phi4Multimodal(BaseModel):
    def __init__(self, llm_path='microsoft/Phi-4-multimodal-instruct'):
        kwargs = {}
        kwargs['torch_dtype'] = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(llm_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='eager',
        ).cuda()

        self.generation_config = GenerationConfig.from_pretrained(llm_path)

        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
        print(self.generation_config)

    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=1024,
    ):  
        prompt = f'{self.user_prompt}<|audio_1|>{prompt}{self.prompt_suffix}{self.assistant_prompt}'
        print(f'>>> Prompt\n{prompt}')
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors='pt').to('cuda')
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f'>>> Response\n{response}')
        return response


    def chat_mode(self, audio, sr, max_new_tokens=1024):
        chat = [{'role': 'user', 'content': f'<|audio_1|>'}]
        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
        audio = (audio, sr)
        inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_logits_to_keep=0,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response