PROMPT = [
    "Transcribe this speech into text.",
    "Convert this audio to text.",
    "Write down the spoken words here.",
    "Speech to text for this recording.",
    "Get me the transcript for this audio.",
    "Put this voice input into words.",
    "What was said? Write it down.",
    "Generate text from this voice data.",
    "Turn this sound bite into a script.",
    "Provide the written form of this speech.",
    "Listen and type out the speech.",
    "Audio to text conversion required.",
    "Transcribe speech to text.", # Direct, common
    "Need a text script generated from the input speech sound.", # Concise need
    "Obtain the textual content corresponding to this voice message.", # Formal action verb (obtain)
    "Text transcription needed urgently for this audio.", # Urgency
    "Can you type up exactly what's said in this recording?", # Informal question
    "Provide the text equivalent of this spoken message ASAP.", # Urgency, different phrasing
    "Just write down the words from the audio.", # Very direct, simple
    "Convert voice recording into written words.", # Simple, direct verb
    "Could you please transcribe the following audio recording into written text format?", # Polite request
    "I need you to listen to this speech carefully and provide me with a written transcript.", # Specific instruction
    "Generate the text that corresponds exactly to the words spoken in this audio clip.", # Focus on exactness
    "Would you mind turning this voice message I have into a text document for me?", # Polite, specific input type
    "Provide a verbatim transcription of the speech which is contained within this audio input.", # Explicitly verbatim
    "Can you process this sound file I'm providing and give me back the text version?", # Question format, common terms
    "I require a complete written record of the conversation that was captured in this audio.", # Formal requirement
    "Please perform speech-to-text processing on the provided sound input.", # Technical phrasing
    "Convert the utterances made in this audio segment into their proper textual representation.", # Different vocabulary (utterances, representation)
    "I want you to generate the full script based on this spoken audio track I have.", # Focus on 'script' output
    "Turn every single spoken word in this file into the corresponding written text equivalent.", # Emphasis on completeness
    "Could you possibly render this spoken language passage into a standard written format?", # Different verb (render), formal
    "Your assigned task: accurately transcribe the accompanying audio material into plain text.", # Direct task assignment
    "I'm looking for a service to convert this speech recording into text.", # Stating user need/search
    "Convert the voice data here into a textual representation for analysis.", # Specific purpose (analysis)
    "Write down everything said in this audio clip, word for word.", # Emphasis on verbatim
    "I need this audio file turned into words on a page, thanks.", # Informal, common phrasing
    "I have an important audio recording of a meeting, and I need you to carefully transcribe every single word spoken into a standard text document for my later review.", # Context (meeting), detail (carefully, every word), purpose (review)
    "Could you please apply your advanced speech recognition capabilities to this lengthy audio file and generate a complete and highly accurate textual transcript of its entire contents?", # Focus on capability, accuracy, length
    "For official documentation purposes, I require a verbatim text output that meticulously captures all the spoken elements present in the provided voice recording file.", # Formal purpose, detail (meticulously)
    "Listen very closely to the following sound clip and meticulously document every single word uttered by the speaker, producing a comprehensive written transcript when you are completely finished.", # Detailed instruction, focus on speaker
    "Please process the enclosed audio data stream and return the corresponding transcribed text, ensuring the highest possible fidelity to the original spoken words.", # Technical input (stream), quality requirement (fidelity)
    "The goal here is straightforward: take this speech input file, analyze the spoken language components, and produce the equivalent written text output as accurately as possible.", # Explaining the goal, process steps
    "This audio contains critical information; I need a flawless transcription converting all spoken dialogue into a text format I can easily share and reference.", # High importance, quality (flawless), usability (share, reference)
    "Can your system handle transcribing this rather noisy audio recording? I need the best possible text representation of the speech, despite the background interference.", # Addressing challenge (noise)
    "I'm providing a voice memo, and my explicit instruction is for you to generate an accurate, word-for-word text transcription of everything said within it.", # Specific input type, explicit instruction
    "Please take the raw audio feed provided and transform it into a structured text document containing only the transcribed speech, potentially omitting non-speech sounds.", # Technical input (feed), specific output format (structured), feature (omit non-speech)
    "For accessibility reasons, could you generate a clean text transcript of the spoken content in this audio file, ensuring readability and accuracy?", # Specific purpose (accessibility)
    "Analyze the speech within this file and furnish me with a complete transcription, paying close attention to speaker changes if discernible.", # Focus on analysis, speaker diarization aspect
    "Turn this audio recording of a lecture into a properly formatted text transcript that I can use for studying and reference purposes later on.", # Specific context (lecture), purpose (studying)
]

ASR_SHORT_PROMPT = [
    # Short (approx. <=10 words)
]

ASR_MEDIUM_PROMPT = [
    # Medium (approx. 11-20 words) - Keeping varied phrasing, requests, instructions
]

ASR_LONGER_PROMPT = [
    # Longer (approx. 20-30 words) - Keeping diverse scenarios, requirements, details
]


import os
import json
import argparse

def classify_instruction(instruction: str) -> str:
    """根据instruction内容返回分类"""
    if instruction in ASR_SHORT_PROMPT:
        return "short"
    elif instruction in ASR_MEDIUM_PROMPT:
        return "medium"
    elif instruction in ASR_LONGER_PROMPT:
        return "longer"
    else:
        raise (KeyError, instruction) 
    
with open("../data/dataset/librispeech_audios.txt", 'r') as f:
    librispeech_texts = f.readlines()
    librispeech_texts = [l.rstrip('\n') for l in librispeech_texts]
with open("../data/dataset/commonvoice_audios.txt", 'r') as f:
    commonvoice_texts = f.readlines()
    commonvoice_texts = [l.rstrip('\n') for l in commonvoice_texts]
    
def classify_dataset(target: str) -> str:
    """根据targer内容返回数据集名称"""
    if target in librispeech_texts:
        return "librispeech"
    elif target in commonvoice_texts:
        return "commonvoice"
    else:
        raise (KeyError, target) 

def process_json_files(input_dir: str, output_dir: str):
    """处理目录中的所有JSON文件"""
    # 创建输出目录
    
    for dataset_name in ["librispeech", "commonvoice"]:
        os.makedirs(os.path.join(output_dir, dataset_name, "short"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, dataset_name, "medium"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, dataset_name, "longer"), exist_ok=True)
        
    # 遍历输入目录中的所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 初始化分类字典
                classified = {
                    "librispeech": {
                        "short": [],
                        "medium": [],
                        "longer": []
                    },
                    "commonvoice": {
                        "short": [],
                        "medium": [],
                        "longer": []
                    }
                }
                
                # 分类每个字典
                for item in data:
                    category = classify_instruction(item["prompt"])
                    dataset_name = classify_dataset(item["target"])
                    classified[dataset_name][category].append(item)
                
                # 保存分类结果
                for dataset, results in classified.items():
                    for category, items in results.items():
                        output_path = os.path.join(output_dir, dataset, category, filename)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(items, f, ensure_ascii=False, indent=2)
                
                print(f"处理完成: {filename}")
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    for item in PROMPT:
        l = len(item.split())
        if l <= 10:
            ASR_SHORT_PROMPT.append(item)
        elif l <=20:
            ASR_MEDIUM_PROMPT.append(item)
        else:
            ASR_LONGER_PROMPT.append(item)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", '-i', type=str, required=True, help="输入JSON文件目录")
    parser.add_argument("--output_dir", '-o', type=str, default='../data/classified', help="输出目录")
    args = parser.parse_args()
    
    process_json_files(args.input_dir, args.output_dir)

