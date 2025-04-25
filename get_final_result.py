import json

def merge_json_with_min_values(file1_path, file2_path, output_path):
    # 读取第一个JSON文件
    with open(file1_path, 'r') as f1:
        data1 = json.load(f1)
    
    # 读取第二个JSON文件
    with open(file2_path, 'r') as f2:
        data2 = json.load(f2)
    
    # 递归合并函数
    def merge_dicts(d1, d2):
        merged = {}
        for key in d1:
            if key in d2:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merged[key] = merge_dicts(d1[key], d2[key])
                elif isinstance(d1[key], (int, float)) and isinstance(d2[key], (int, float)):
                    merged[key] = min(d1[key], d2[key])
                else:
                    # 如果不是数字类型，保留第一个文件的值
                    raise ValueError
                    # merged[key] = d1[key]
            else:
                merged[key] = d1[key]
        
        # 添加d2中独有的键
        for key in d2:
            if key not in d1:
                merged[key] = d2[key]
        
        return merged
    
    # 执行合并
    merged_data = merge_dicts(data1, data2)
    
    # 写入输出文件
    with open(output_path, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)
    
    print(f"合并完成，结果已保存到 {output_path}")

# 使用示例
merge_json_with_min_values(
    file1_path='../data/results_aligned.json',
    file2_path='../data/results_pred.json',
    output_path='../data/results_final.json'
)