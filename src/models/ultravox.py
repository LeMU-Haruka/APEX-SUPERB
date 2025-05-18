import transformers

from src.models.base_model import BaseModel


class Ultralvox(BaseModel):
    def __init__(self, llm_path='fixie-ai/ultravox-v0_5-llama-3_1-8b'):
        self.pipe = transformers.pipeline(model=llm_path, trust_remote_code=True)

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        assert sr == 16000
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
        ]
        response = self.pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=max_new_tokens)
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=2048,
    ):
        assert sr == 16000
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
                        {
                "role": "user",
                "content": prompt
            },
        ]
        response = self.pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=max_new_tokens)
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