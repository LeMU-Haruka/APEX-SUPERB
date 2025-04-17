#!/bin/bash 


# 检查输入参数是否齐全：model, model_path, gpu_id, group_id
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <model> <model_path> <gpu_id> <group_id (0-3)>"
  exit 1
fi

model=$1
model_path=$2
gpu_id=$3
group=$4


num_groups=2
# 判断 group 参数是否在 num_groups 范围内
if ! [[ "$group" =~ ^[0-9]+$ ]] || [ "$group" -ge "$num_groups" ] || [ "$group" -lt 0 ]; then
  echo "错误：group 参数必须在 0 到 $((num_groups - 1)) 之间"
  exit 1
fi



# 设置 CUDA_VISIBLE_DEVICES 环境变量，确保脚本只使用指定的 GPU
export CUDA_VISIBLE_DEVICES=5,$gpu_id
echo "[$CUDA_VISIBLE_DEVICES]"

# 定义你要遍历的 task 列表
tasks=(
    # instruction tasks
    "text_instruct_st"
    "ifeval"
    "gsm8k_fewshot_1"
    "gsm8k_fewshot_2"
    "gsm8k_fewshot_4"
    "gsm8k_fewshot_8"

    # QA and reasoning tasks
    "gsm8k"
    "alpaca_empathy"
    "mmlu"
    "alpaca_eval"
    "speaker_role"
)

total=${#tasks[@]}


# 计算每组的基本大小和余数，以实现尽可能均衡的拆分
base=$(( total / num_groups ))
rem=$(( total % num_groups ))


if [ "$group" -lt "$rem" ]; then
  group_size=$(( base + 1 ))
  start_index=$(( group * (base + 1) ))
else
  group_size=$base
  start_index=$(( rem * (base + 1) + (group - rem) * base ))
fi

end_index=$(( start_index + group_size ))

echo "总任务数：$total"
echo "将任务拆分为 $num_groups 组"
echo "当前选取组号：$group, 任务索引范围：[$start_index, $end_index)"
echo "本组任务数量：$group_size"


start_time=$(date +%s)

# 遍历任务
success_tasks=()
failed_tasks=()

for (( i = start_index; i < end_index; i++ )); do
  task=${tasks[$i]}
  echo "【运行】任务：$task 在 GPU $gpu_id 上"
  if python generate.py --model "$model" --model_path "$model_path" --task "$task"; then
    success_tasks+=("$task")
  else
    failed_tasks+=("$task")
  fi
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# 输出用时信息
echo "All Done!"
echo "Succeeded tasks: ${success_tasks[*]}"
echo "Failed tasks: ${failed_tasks[*]}"
echo "Total time taken: $elapsed_time seconds"