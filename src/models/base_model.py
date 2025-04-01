import torch


class BaseModel:
    @torch.no_grad()
    def chat_mode(
            self,
            audio,
            max_new_tokens=2048,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def prompt_mode(
            self,
            prompt,
            audio,
            max_new_tokens=2048,
    ):
        raise NotImplementedError

    def text_mode(self, prompt, text, max_new_tokens=2048):
        raise NotImplementedError
