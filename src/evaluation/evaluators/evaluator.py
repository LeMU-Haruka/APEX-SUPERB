

class Evaluator:

    def evaluate(self, data):
        raise NotImplementedError

    def align_text(self, item):
        pred = item['pred']
        prompt = self.align_prompt.replace("[INPUT]", pred)
        response = self.client.generate_response(prompt)
        self.cache_file.append({
            'pred': pred,
            'aligned_pred': response
        })
        return response
    
    # def align_text(self, item):
    #     align_prompt = """
    #         You are a text processing assistant. Your task is to clean the provided text by removing any extraneous, redundant, or non-essential expressions while preserving the core semantic content. This includes eliminating introductory statements, irrelevant formatting elements, unnecessary punctuation, or any additional commentary that does not affect the meaning.

    #         For example, if given the input:
    #         -------------------------------------------------
    #         The original content of this audio is: 'Yesterday you were trembling for a health that is dear to you, today you fear for your own, tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer, the day after that the misfortune of some friend, then the prevailing weather, then something that has been broken or lost, then a pleasure with which your conscience and your vertebral column rebel.
    #         -------------------------------------------------
    #         The expected cleaned output should be:
    #         -------------------------------------------------
    #         yesterday you were trembling for a health that is dear to you to day you fear for your own to morrow it will be anxiety about money the day after to morrow the diatribe of a slanderer the day after that the misfortune of some friend then the prevailing weather then something that has been broken or lost then a pleasure with which your conscience and your vertebral column reproach you again the course of public affairs
    #         -------------------------------------------------

    #         This prompt should be applicable in all cases—whether the task involves translation, processing multiple-choice options, or any similar scenario where extra expressions are present. Only output the cleaned text.

    #         Now, please process the following text:
    #         -------------------------------------------------
    #         [INPUT]
    #         -------------------------------------------------
    #         The Output should only contain the cleaned text.
    #         """
    #     text = item['question']
    #     align_prompt = align_prompt.replace("[INPUT]", text)
    #     response = self.generate_response(align_prompt)
    #     aligned_text = response.text
    #     return aligned_text