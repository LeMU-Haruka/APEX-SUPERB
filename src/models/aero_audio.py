from transformers import AutoModelForCausalLM, AutoProcessor

from src.models.base_model import BaseModel


class AeroAudio(BaseModel):
    def __init__(self, llm_path='lmms-lab/Aero-1-Audio'):
        self.processor = AutoProcessor.from_pretrained(llm_path, trust_remote_code=True)
        # We encourage to use flash attention 2 for better performance
        # Please install it with `pip install --no-build-isolation flash-attn`
        # If you do not want flash attn, please use sdpa or eager`
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="cuda", torch_dtype="auto", attn_implementation="sdpa", trust_remote_code=True)
        self.model.eval()

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        assert sr == 16000
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio": "placeholder",
                    }
                ]
            }
        ]
        audios = [audio]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, audios=audios, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, eos_token_id=151645, max_new_tokens=4096)

        cont = outputs[:, inputs["input_ids"].shape[-1] :]

        response = self.processor.batch_decode(cont, skip_special_tokens=True)[0]
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=2048,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio": "placeholder",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
        audios = [audio]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, audios=audios, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, eos_token_id=151645, max_new_tokens=4096)

        cont = outputs[:, inputs["input_ids"].shape[-1] :]

        response = self.processor.batch_decode(cont, skip_special_tokens=True)[0]
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