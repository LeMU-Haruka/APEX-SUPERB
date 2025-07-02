import os

import numpy as np
from typing import  Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import lightning.pytorch as pl

import logging

from torch.nn.utils.rnn import pad_sequence
from transformers import   AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, \
    Wav2Vec2FeatureExtractor, LlamaForCausalLM, BertTokenizer, BertModel

from src.models.mtbi_modules.modeling_adapter import Subsampler
# from modeling_llama import LlamaForCausalLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INSTRUCTION_MAP = {
    'asr': 'Please Transcribe this speech: ',
    'continue': 'Continue the following speech in a coherent and engaging style with less than 100 words:',
    'select_v': 'Select all the verb in the following speech and return them in a list: ',
    'select_n': 'Select all the noun in the following speech and return them in a list: ',
    'rewrite': 'Rewrite the following speech while maintaining its original meaning with less than 100 words: '
}


class MTBI(pl.LightningModule):
    def __init__(
        self,
        llama_hidden_size=4096,
        speech_model_path="/userhome/models/wavlm-large",
        # llama_ckpt="/userhome/models/lama-2-7b-chat-hf"):
        llama_ckpt="/userhome/models/Llama3.1-8B-Instruct",):
        super(MTBI,self).__init__()

        #path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        speech_model_path = os.path.join(current_directory, speech_model_path)
        llama_ckpt = os.path.join(current_directory, llama_ckpt)

        #hubert
        self.speech_encoder = AutoModel.from_pretrained(speech_model_path)
        self.speech_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_path)
        self.layer_num = 25
        #llama
        self.llama_model=AutoModelForCausalLM.from_pretrained(llama_ckpt, torch_dtype="auto")
        #self.llama_model = self.llama_model.to(torch.float32)
        self.llama_tokenizer=AutoTokenizer.from_pretrained(llama_ckpt)
        # if hasattr(torch, 'compile'):
        #     print("Compiling the model with torch.compile()...")
        #     self.llama_model = torch.compile(self.llama_model)

        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = 'left' 
        # self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for p in self.speech_encoder.parameters():
            p.requires_grad = False
        for p in self.llama_model.parameters():
            p.requires_grad = False
        self.speech_connector = Subsampler(1024, 768, llama_hidden_size,
                                      0, "5,5,5")
        self.print_count = 0 


    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



    def forward(self, audio, labels, task):
        batch_size = len(audio)
        self.print_count += 1
        inputs = self.speech_feature_extractor(
            audio, padding=True, return_tensors="pt", sampling_rate=16000
        ).to(self.device)
        with torch.no_grad():
            # 一次性处理整个批次
            audio_hidden_states = self.speech_encoder(inputs.input_values.float()).last_hidden_state

        attention_mask = self.speech_encoder._get_feature_vector_attention_mask(
            audio_hidden_states.shape[1], inputs.attention_mask
        )
        audio_input, audio_atts, _ , _,_= self.speech_connector(audio_hidden_states, attention_mask)
        audio_input = audio_input.to(dtype=self.llama_model.dtype) 

        instructions = [INSTRUCTION_MAP[t] for t in task]
        prefix_ids, prefix_atts = self.build_prompt(instructions)
        prefix_ids, prefix_atts = prefix_ids.to(self.device), prefix_atts.to(self.device)

        # 3. 构建模型需要学习的目标部分 (Target Label)
        # 目标包含了完整的助手回合：结束用户回合的<|eot_id|> + 助手前缀 + 回答文本 + 结束助手回合的<|eot_id|>
        assistant_turn_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        
        # 清理用户提供的 label，去除可能的前导空白
        clean_answers = [s.lstrip().lower() for s in labels]
        
        # 将清理后的回答填入模板
        target_texts = [assistant_turn_template.format(answer=ans) for ans in clean_answers]


        # 对完整的助手回合进行分词
        # 注意：这里的 padding_side 应该是 right，因为这是模型要生成的部分
        self.llama_tokenizer.padding_side = 'right'
        target_tokens = self.llama_tokenizer(
            target_texts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True, 
            add_special_tokens=False
        )
        self.llama_tokenizer.padding_side = 'left' # 恢复为默认的生成设置

        target_ids = target_tokens.input_ids.to(self.device)
        target_atts = target_tokens.attention_mask.to(self.device)

        # 4. 将 Input 和 Target 拼接成完整的序列，用于一次前向传播
        # 获取各部分 Embedding
        prefix_embeds = self.llama_model.model.embed_tokens(prefix_ids)
        target_embeds = self.llama_model.model.embed_tokens(target_ids)

        input_embeds = torch.cat([prefix_embeds, audio_input, target_embeds], dim=1)
        # 拼接 Attention Mask
        attention_mask = torch.cat([prefix_atts, audio_atts, target_atts], dim=1)
        
        # 5. 构建用于计算 Loss 的 Labels 张量
        # 我们需要屏蔽掉所有输入部分 (Prefix + Audio)，只在目标部分 (Target) 计算损失
        prefix_len = prefix_embeds.shape[1]
        speech_len = audio_input.shape[1]
        
        # 创建一个全为 -100 的 labels 张量
        labels_tensor = torch.full(attention_mask.shape, -100, dtype=torch.long, device=self.device)
        
        # 将目标部分的 token ID 复制到正确的位置
        # 同时，将目标部分中的 padding token 也设置为 -100
        masked_target_ids = target_ids.masked_fill(target_ids == self.llama_tokenizer.pad_token_id, -100)
        labels_tensor[:, (prefix_len + speech_len):] = masked_target_ids
        
        # 6. 调用模型，使用 Hugging Face 内置的 Loss 计算
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels_tensor, # 传入 labels，模型会自动移位并计算 loss
            return_dict=True
        )
        
        loss = outputs.loss

        # 每100step 打印一下response
        if self.print_count % 100 == 0:
            # self.trainer.is_global_zero 确保只在主进程（rank 0）中打印，避免多GPU时重复打印
            
            # 进入评估模式，这会禁用 dropout 等，确保推理行为的一致性
            self.eval() 
            with torch.no_grad():
                # 我们只取 batch 中的第一个样本进行推理演示
                sample_audio = [audio[0]]
                sample_label = labels[0]
                sample_task = task[0]
                
                # 调用修正后的 inference 方法
                generated_text_list = self.inference(audio=sample_audio, task=sample_task)
                generated_text = generated_text_list[0] if generated_text_list else "Generation failed"

            # 打印结果
            print("\n" + "="*20 + f" DEBUG (Step: {self.global_step}) " + "="*20)
            print(f"  Task: {sample_task}")
            print(f"  Target: {sample_label.lstrip().lower()}")
            print(f"  Generated: {generated_text}")
            print("="*58 + "\n")
            
            # 恢复到训练模式
            self.train() 

        return loss

    # def sample_negative(self, audio, task):
        

    def build_prompt(self, instructions: list):
        """
        [最终版本]
        精确地从 Llama 3 模板中分割出音频内容（content）前后的部分。
        这个函数只负责构建模型的“输入”部分。
        """
        # 定义模板中，代表“用户”回合的部分，直到内容占位符之前
        prompt_before_content = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction} "  # 注意末尾的空格，它将与音频嵌入或文本内容分隔
        )
        
        # 填充指令，创建完整的批量前缀文本
        prefix_texts = [prompt_before_content.format(instruction=inst) for inst in instructions]

        # [核心] 使用 tokenizer 的批量处理能力，它会自动处理左填充
        prefix_inputs = self.llama_tokenizer(
            prefix_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # 返回分词后的 token ID 和对应的 attention_mask
        return prefix_inputs.input_ids, prefix_inputs.attention_mask

    def training_step(self, batch, batch_idx):
        audio = batch['wav']
        label = batch['label']
        task = batch['task']
        loss = self.forward(audio, label, task)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(audio),
                 sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch['wav']
        label = batch['label']
        task = batch['task']
        loss = self.forward(audio, label, task)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(audio),
                 sync_dist=True)
        # with torch.no_grad():
        #     output = self.inference(audio)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.000013,
                                      betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer

    def inference(self, audio: list, task: str, prompt: str = None):
        """
        [修正后]
        根据音频输入生成文本，确保输入格式与训练过程完全一致。
        """
        # ===================================================================
        # 1. 音频处理 (与训练时完全一样)
        # ===================================================================
        inputs = self.speech_feature_extractor(
            audio, padding=True, return_tensors="pt", sampling_rate=16000
        ).to(self.device)
        with torch.no_grad():
            audio_hidden_states = self.speech_encoder(inputs.input_values.float()).last_hidden_state
        attention_mask = self.speech_encoder._get_feature_vector_attention_mask(
            audio_hidden_states.shape[1], inputs.attention_mask
        )
        audio_input, audio_atts, _, _, _ = self.speech_connector(audio_hidden_states, attention_mask)
        batch_size = audio_input.shape[0]

        # ===================================================================
        # 2. Prompt 构建 (与训练时完全一样)
        # ===================================================================
        if prompt is None:
            instruction = INSTRUCTION_MAP[task]
        else:
            instruction = prompt
        
        # 使用与训练完全相同的 build_prompt 函数
        # 注意：需要为批次中的每个样本都提供指令
        prefix_ids, prefix_atts = self.build_prompt([instruction] * batch_size)
        prefix_ids, prefix_atts = prefix_ids.to(self.device), prefix_atts.to(self.device)

        # 准备模型生成前的完整输入
        prefix_embeds = self.llama_model.model.embed_tokens(prefix_ids)
        
        # [核心] 推理时的输入只包含 prefix 和 audio
        inputs_embeds = torch.cat([prefix_embeds, audio_input], dim=1)
        attention_mask_for_generation = torch.cat([prefix_atts, audio_atts], dim=1)
        
        # ===================================================================
        # 3. 触发助手回答的 Token
        # ===================================================================
        # 我们需要手动创建 "assistant" header 的 token，并将其作为生成的起点
        assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_prompt_ids = self.llama_tokenizer(
            assistant_prompt, 
            add_special_tokens=False, 
            return_tensors='pt'
        ).input_ids.to(self.device)
        
        # ===================================================================
        # 4. 模型生成
        # ===================================================================
        # 定义停止符，这对于 Llama 3 至关重要
        terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_for_generation,
                # 使用 assistant_prompt_ids 作为 bos_token_id 的一种 "hack" 方式来强制开头
                # 或者更标准的方式是在生成后处理
                # 这里我们采用更简单的方式，让模型自由生成，然后后处理
                max_new_tokens=300,
                eos_token_id=terminators,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                pad_token_id=self.llama_tokenizer.eos_token_id # 推荐
            )

        # ===================================================================
        # 5. 解码和后处理
        # ===================================================================
        # 从生成结果中去除输入 prompt 的部分
        output_tokens = outputs[:, inputs_embeds.shape[1]:]
        raw_outputs = self.llama_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        
        # 清理 assistant 前缀
        clean_results = []
        assistant_prefix = "assistant\n\n"
        for raw_output in raw_outputs:
            processed_output = raw_output.lstrip()
            if processed_output.startswith(assistant_prefix):
                result = processed_output[len(assistant_prefix):]
            else:
                result = processed_output
            clean_results.append(result)
            
        return clean_results


    def generate(self, audio: list, prompt: str = None, new_max_tokens: int = 200):
        """
        [修正后]
        根据音频输入生成文本，确保输入格式与训练过程完全一致。
        """
        # ===================================================================
        # 1. 音频处理 (与训练时完全一样)
        # ===================================================================
        inputs = self.speech_feature_extractor(
            audio, padding=True, return_tensors="pt", sampling_rate=16000
        ).to(self.device)
        with torch.no_grad():
            audio_hidden_states = self.speech_encoder(inputs.input_values.float()).last_hidden_state
        attention_mask = self.speech_encoder._get_feature_vector_attention_mask(
            audio_hidden_states.shape[1], inputs.attention_mask
        )
        audio_input, audio_atts, _, _, _ = self.speech_connector(audio_hidden_states, attention_mask)
        batch_size = audio_input.shape[0]


        # ===================================================================
        # 2. Prompt 构建 (与训练时完全一样)
        # ===================================================================
        if prompt is None:
            instruction = ''
        else:
            instruction = prompt
        
        # 使用与训练完全相同的 build_prompt 函数
        # 注意：需要为批次中的每个样本都提供指令
        prefix_ids, prefix_atts = self.build_prompt([instruction] * batch_size)
        prefix_ids, prefix_atts = prefix_ids.to(self.device), prefix_atts.to(self.device)

        # 准备模型生成前的完整输入
        prefix_embeds = self.llama_model.model.embed_tokens(prefix_ids)
        assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_prompt_ids = self.llama_tokenizer(
            [assistant_prompt] * batch_size,
            add_special_tokens=False, 
            return_tensors='pt'
        ).input_ids.to(self.device)
        assistant_prompt_embeds = self.llama_model.model.embed_tokens(assistant_prompt_ids)
        assistant_prompt_atts = torch.ones_like(assistant_prompt_ids) # attention mask 全为 1
        inputs_embeds = torch.cat([prefix_embeds, audio_input, assistant_prompt_embeds], dim=1)
        attention_mask_for_generation = torch.cat([prefix_atts, audio_atts, assistant_prompt_atts], dim=1)

        # ===================================================================
        # 4. 模型生成
        # ===================================================================
        # 定义停止符，这对于 Llama 3 至关重要
        terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # ===================================================================
        # 4. 模型生成
        # ===================================================================
        # 定义停止符，这对于 Llama 3 至关重要
        terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_for_generation,
                max_new_tokens=200,
                eos_token_id=terminators,
                do_sample=True,
                top_p=0.8,
                temperature=0.5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.llama_tokenizer.eos_token_id # 推荐
            )

        # ===================================================================
        # 5. 解码和后处理
        # ===================================================================
        # 从生成结果中去除输入 prompt 的部分
        output_tokens = outputs[:, inputs_embeds.shape[1]:]
        raw_outputs = self.llama_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        # 只需要去除可能的前导空格即可
        clean_results = [s.lstrip() for s in raw_outputs]
        return clean_results