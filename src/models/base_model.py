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
        audio = item['audio']
        sr = item['sr']
        prompt = item['instruction']
        if task == 'speech_instruct_asr':
            pred = self.chat_mode(audio, sr)
        else:
            pred = self.prompt_mode(prompt, audio, sr)
        return {
            'file': item['file'],
            'prompt': prompt,
            'question': item['text'],
            'pred': pred,
            'target': item['label'],
            'kargs': item['kargs'],
        }