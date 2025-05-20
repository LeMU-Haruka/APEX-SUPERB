# APEX-SUPERB

APEX-SUPERB is a holistic ability-oriented benchmark that evaluates Audio LLMs across nine core abilities under three interaction pillars, providing comprehensive and user-friendly performance analysis beyond narrow task metrics.

## 🌟 Features

- **Comprehensive Evaluation**: Tests 9 core abilities across 3 interaction pillars
- **Multiple Model Support**: 
  - Closed-source models (Gemini, GPT)
  - Open-source ALLMs (10+ models)
  - Cascaded models
- **Flexible Framework**: Easy integration of new models and tasks
- **Standardized Metrics**: Consistent evaluation across different model types

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/APEX-SUPERB.git
cd APEX-SUPERB
```

2. Create and activate a virtual environment (recommended):
We highly recommend using **Python version ≤ 3.11**, as some modules still rely on the deprecated `distutils` module, which has been removed in Python 3.12. We are actively monitoring module compatibility and will provide updates as needed.

```bash
# Using conda (recommended)
conda create -n apex python=3.11
conda activate apex

# Or using venv
python3.11 -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
Due to the involvement of multiple models with varying requirements, we only provide a minimal `requirements.txt` containing key dependencies for the core modules. This helps keep the environment clean and avoids unnecessary conflicts. 

You can install model-specific dependencies by referring to the documentation on each model’s GitHub repository or Hugging Face page.

To install the core dependencies:

```bash
pip install -r requirements.txt
```

> **Important Note on Model Dependencies**  
> Due to version conflicts in modules such as Hugging Face Transformers, some models may require different versions of dependencies. We recommend:
>
> - Commenting out conflicting models in `src/models/__init__.py` to avoid blocking other model tests  
> - Creating separate virtual environments for models with conflicting dependencies  
> - Installing specific versions of dependencies for each model as needed  


## 🔧 Data Preparation

Download all necessary datasets for evaluation using our preparation script:

```bash
python src/datasets/prepare.py
```

This will download all datasets to `./local_datasets/`.

## 🚀 Running Evaluations

### 1. Generate Responses

You can evaluate models in two ways:

#### Option 1: Using one line commands
```bash
./generate.sh {model_name} {model_path} {gpu} {split}
```

#### Option 2: Using Python Directly
```bash
python generate.py --model {model_name} --model_path {model_path} --task {task_name}
```

Available models include:
- Closed-source: `gemini`, `gpt4`
- Open-source: `whisper`, `qwen2`, `salmonn`, etc.
- Cascaded: `cascaded_llama3`, `cascaded_qwen2`
All model supported are in model register [file](src/models/__init__.py).

### 2. Evaluate Results

Run the evaluation script to compute metrics:

```bash
python evaluate.py --model {model_name} --result_path {result_path} --api {api} --align
```

Parameters:
- `--model`: Name of the model to evaluate
- `--result_path`: Path to the generated results
- `--api`: API to use for evaluation (`gemini`, `gpt`, `vllm`)
- `--align`: Enable LLM-based alignment for ASR and speech command tasks

For specific task evaluation:
```bash
python evaluate.py --model {model_name} --result_path {result_file_path} --task {task_name}
```

## 🔄 Adding New Models

We provide a flexible framework for integrating new models. See our [Model Integration Guide](src/models/README.md) for detailed instructions on:
- Implementing the required interfaces
- Handling audio processing
- Managing memory and optimization
- Testing your integration

## 📊 Tasks and Metrics

APEX-SUPERB evaluates models across various tasks:
- Automatic Speech Recognition (ASR)
- Speech Command Recognition
- Audio Question Answering
- Audio Captioning
- And more...

Each task uses appropriate metrics for evaluation, with LLM-based alignment available for certain tasks to handle model-specific output variations.

## 🛠️ Customizing Tasks

Our task dataset follows the same format as [DYNAMIC-SUPERB](https://github.com/dynamic-superb/dynamic-superb/blob/main/docs/task_submission.md). You can:
1. Use Hugging Face datasets directly by specifying the repo name as `{task_name}`
2. Save local data in the `local_datasets` directory using the same structure as `{task_name}`

Therefore, all tasks in [DYNAMIC-SUPERB](https://huggingface.co/DynamicSuperb) can directly run on our framework.

## 📝 Citation

```bibtex
[Citation will be added after paper publication]
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

<!-- ## 📄 License

[License information will be added] -->

## 📬 Contact

For questions or issues:
1. Create an issue in this repository
2. Check existing model implementations for reference
3. Follow our code style guidelines

## News
- *May 15, 2025*: We open source the evaluation code, and [dataset](https://huggingface.co/APEX-SUPERB).

## Add Your Own Models
We provide a [guide](src/models/README.md) on how to integreted a model in our framework. 

## Construct Your Own Tasks
Our task dataset is following the same manner as [DYNAMIC-SUPERB] (https://github.com/dynamic-superb/dynamic-superb/blob/main/docs/task_submission.md). We support huggingface dataset. You can directly put your HF repo in {task_name} or save the local data into local_datasets directoy and use the same directory name as {task_name}.

## Paper and Documentation
The paper is currently under review, will release soon.