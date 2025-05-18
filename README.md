# APEX-SUPERB

APEX-SUPERB is a holistic ability-oriented benchmark that evaluates Audio LLMs across nine core abilities under three interaction pillars, providing comprehensive and user-friendly performance analysis beyond narrow task metrics.

## News
- *May 15, 2025*: We open source the evaluation code, and [dataset](https://huggingface.co/APEX-SUPERB).

## Data Preparation
We provide the python script to quick download all the neccesary dataset for evaluation. After run the code below, all dataset will downloaded under the APEX-SUPERB/local_datasets.

```shell
python src/datasets/prepare.py
```

## Gnerate Response
We implement lots of Audio LLMs in our framework, including 2 types of closed-source models (Gemini, GPT), cascaded models, and 10 open-source ALLMs, that can directly run with the code.
The result will saved in the {model_name}_results

```shell
./generate.sh {model_name} {model_path} {gpu} {split}
```

You can also use following command to run specific tasks.

```shell
python generate.py --model {model_name} --model_path {model_path} --task {task_name}
```

## Evaluation
After generate the output results, we can use one line command to evaluation the score. With --align command, we will do LLM-based align for Automatic speech recognition and speech command tasks, since some model will output large amount of rebundant content in the response, making these two metric very low.

```shell
python evaluate.py --model {model_name} --result_path {result_path} --api {api} --align 
```

You can also use following command to run on specific results.

```shell
python evaluate.py --model {model_name} --result_path {result_file_path} --task {task_name}
```


## Add Your Own Models
We provide a [guide](src/models/README.md) on how to integreted a model in our framework. 


## Construct Your Own Tasks
Our task dataset is following the same manner as [DYNAMIC-SUPERB] (https://github.com/dynamic-superb/dynamic-superb/blob/main/docs/task_submission.md). We support huggingface dataset. You can directly put your HF repo in {task_name} or save the local data into local_datasets directoy and use the same directory name as {task_name}.




## Paper and Documentation
The paper is currently under review, will release soon.