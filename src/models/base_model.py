import torch


class BaseModel:
    @torch.no_grad()
    def chat_mode(
            self,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def prompt_mode(
            self,
            instruction,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        raise NotImplementedError

    def text_mode(self, instruction, text, max_new_tokens=1024):
        raise NotImplementedError


    def process(self, item, task):
        raise NotImplementedError