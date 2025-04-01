from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LLAMA_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"


class LLaMA3:
    def __init__(self, device, is_gt=False):
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def chat_mode(self, text, max_new_tokens=2048):
        conversation = [
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_length=max_new_tokens
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def prompt_mode(self, prompt, text, max_new_tokens=2048):
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_length=max_new_tokens
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
#
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)
#
# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
#
# outputs = model.generate(
#     input_ids,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True))