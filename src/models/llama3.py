import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaMA3Model:

    def __init__(self, llm_path='meta-llama/Llama-3.1-8B-Instruct' , device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map=device)

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
            {"role": "system",
            "content": "You are a concise, knowledgeable assistant."},
            {"role": "user",
            "content": "### Task Instruction\n" + instruction + "\n\n### Question\n" + text},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        return prompt_ids 

    def process(self, item, task):
        # instruction = item['instruction']
        # text = item['text']
        instruction = 'You are a helpful assistant. Please help to solve the following question.'
        text = item['question']
        pred = self.text_mode(instruction, text)
        item['kargs'] = {}
        item['kargs']['task'] = task
        return {
            # 'file': item['file'],
            'prompt': instruction,
            'question': text,
            'pred': pred,
            'target': item['answer'],
            'kargs': item['kargs'],
        }