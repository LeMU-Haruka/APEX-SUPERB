import os
import json
import argparse

def process_files(input_dir: str, meta_file_path: str, output_dir: str):
    """处理所有JSON文件，更新instruction"""
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json') or filename == os.path.basename(meta_file_path):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i in range(len(data)):
            data[i]["prompt"] = meta_data[i]["instruction"]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, help='包含JSON文件的输入目录')
    parser.add_argument('--meta_file', '-m', required=True, help='包含正确instruction的meta文件')
    parser.add_argument('--output_dir', '-o', default='../data/corrected', help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.meta_file):
        print(f"错误: meta文件 {args.meta_file} 不存在")
        exit(1)
        
    process_files(args.input_dir, args.meta_file, args.output_dir)