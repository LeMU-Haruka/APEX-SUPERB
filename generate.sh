#!/bin/bash 

# 读取输入的 model、model_path 和 GPU ID
model=$1
model_path=$2
gpu_id=$3  # 新增：传入 GPU ID

# 设置 CUDA_VISIBLE_DEVICES 环境变量，确保脚本只使用指定的 GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# 定义你要遍历的 task 列表
tasks=(
  "mmlu"
  "alpaca_eval"
  "asr_librispeech"
  "asr_commonvoice"
  "gsm8k"
  "emotion"
  "dialogue_ser"
  "AED"
  "librispeech_multispeaker"
  "librispeech_emotion"
  "text_instruct_asr"
  "text_instruct_st"
  "empathy"
  "speaker_role"
  "librispeech_speed"
  "ifeval"
  "fewshot_gsm8k_1"
  "fewshot_gsm8k_2"
  "fewshot_gsm8k_4"
  "fewshot_gsm8k_8"
  "librispeech_noise"
  "mmau"
)


start_time=$(date +%s)

# 遍历任务
success_tasks=()
failed_tasks=()

for task in "${tasks[@]}"
do
  echo "Running task: $task on GPU $gpu_id"
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