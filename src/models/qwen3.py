import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.base_model import BaseModel


class Qwen3Model(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen3-8B', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map=device)

    def asr(self, audio, sr):
        assert sr == 16000
        result = self.asr_pipe(audio)
        transcription = result["text"]
        return transcription
    

    def chat_mode(self, audio, sr, max_new_tokens=1024):
        return self.text_mode("You are a helpful voice assistant.", audio, sr, max_new_tokens)
    
    def prompt_mode(self, prompt, audio, sr, max_new_tokens=1024):
        return self.text_mode(prompt, audio, sr, max_new_tokens)

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
                
            
    def text_mode(self, instruction, text, max_new_tokens=1024):
        prompt = self.build_messages(instruction, text)
        out_ids = self.model.generate(
            prompt,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(out_ids[0][prompt.shape[-1]:], skip_special_tokens=True)
        return response


    def build_messages(self, instruction, text):
        messages = [
            {"role": "user", "content": instruction + " " + text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return model_inputs 

    def text_mode(self, prompt, text, max_new_tokens=1024):
        model_inputs = self.build_messages(prompt, text)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print(f"Thinking content: {thinking_content}")
        return content
    
    def process(self, item, task):
        instruction = item['instruction']
        text = item['text']
        pred = self.text_mode(instruction, text)
        item['kargs'] = {}
        item['kargs']['task'] = task
        return {
            'prompt': instruction,
            'question': text,
            'pred': pred,
            'target': item['label'],
            'kargs': item['kargs'],
        } 