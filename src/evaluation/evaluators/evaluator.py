import json
import os


class Evaluator:

    def evaluate(self, data):
        raise NotImplementedError

    def extract_answer(self, item):
        pred = item['pred']
        prompt = self.align_prompt.replace("[INPUT]", pred)
        response = self.client.generate_response(prompt)
        self.cache_file.append({
            'pred': pred,
            'aligned_pred': response
        })
        return response
    

    def save_cache(self, data):
        cache_file = os.path.join(self.cache_dir, f'{self.model_name}_{self.task}.json')
        json_str = json.dumps(data, indent=4)
        with open(cache_file, 'w') as f:
            f.write(json_str)
        print(f"Cache saved to {cache_file}")