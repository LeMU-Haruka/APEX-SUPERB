#!/bin/bash 

# Set task split here
num_groups=4

# Check if all input parameters are provided: model, model_path, gpu_id, group_id
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <model> <model_path> <gpu_id> <group_id (0 - num_groups-1)>"
  exit 1
fi

model=$1
model_path=$2
gpu_id=$3
group=$4

# Check if group parameter is within range 0~3
if ! [[ "$group" =~ ^[0-9]+$ ]] || [ "$group" -ge "$num_groups" ] || [ "$group" -lt 0 ]; then
  echo "Error: group parameter must be between 0 and $((num_groups - 1))"
  exit 1
fi

# Set CUDA_VISIBLE_DEVICES environment variable to ensure the script only uses the specified GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# Define the list of tasks to iterate through
tasks=(
    # basic tasks
    "asr_commonvoice"
    "asr_librispeech"

    "dialogue_ser"
    "emotion_recognition"

    "animal_classification"
    "sound_classification"

    # instruction tasks
    "text_instruct_st"
    "text_instruct_asr"
    "speech_instruct_asr"
    "ifeval"
    "gsm8k_fewshot_1"
    "gsm8k_fewshot_2"
    "gsm8k_fewshot_4"
    "gsm8k_fewshot_8"

    # input robustness tasks
    "librispeech_noise"
    "librispeech_emotion"
    "librispeech_speed"
    "librispeech_multispeaker"

    
    # QA and reasoning tasks
    "gsm8k"
    "alpaca_empathy"
    "mmlu"
    "alpaca_eval"
    "speaker_role"
    "mmau"
)

total=${#tasks[@]}

# Calculate base size and remainder for each group to achieve balanced distribution
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

echo "Total number of tasks: $total"
echo "Tasks divided into $num_groups groups"
echo "Current group number: $group, Task index range: [$start_index, $end_index)"
echo "Number of tasks in this group: $group_size"

start_time=$(date +%s)

# Iterate through tasks
success_tasks=()
failed_tasks=()

for (( i = start_index; i < end_index; i++ )); do
  task=${tasks[$i]}
  echo "【Running】Task: $task on GPU $gpu_id"
  if python generate.py --model "$model" --model_path "$model_path" --task "$task"; then
    success_tasks+=("$task")
  else
    failed_tasks+=("$task")
  fi
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Output time information
echo "All Done!"
echo "Succeeded tasks: ${success_tasks[*]}"
echo "Failed tasks: ${failed_tasks[*]}"
echo "Total time taken: $elapsed_time seconds"