# Model Integration Guide

This guide explains how to integrate your own models into the APEX-SUPERB framework. Our framework supports various types of Audio Language Models (ALLMs) and provides a flexible interface for adding new ones.

## Basic Structure

All models in APEX-SUPERB inherit from the `BaseModel` class, which defines three core interaction modes:

1. `chat_mode`: For open-ended conversations with audio input
2. `prompt_mode`: For task-specific interactions with audio input and text task instructions

## Steps to Add Your Model

### 1. Create a New Model File

Create a new Python file in the `src/models` directory with your model name (e.g., `mymodel.py`).

### 2. Implement the Base Interface

Your model class should inherit from `BaseModel` and implement the following methods:

```python
from src.models.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, model_path):
        # Initialize your model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load your model and other necessary components
        
    def chat_mode(self, audio, sr, max_new_tokens=2048):
        """
        Chat Mode: Generate responses based solely on audio input.

        Args:
            audio (np.ndarray or torch.Tensor): The input audio waveform.
            sr (int): Sampling rate of the audio (typically 16000).
            max_new_tokens (int): Maximum number of tokens to generate. Default is 2048.

        Returns:
            response (str): The generated conversational response.
        """
        return response
        
    def prompt_mode(self, instruction, audio, sr, max_new_tokens=1024):
        """
        Prompt Mode: Perform task-specific interaction based on both text task instruction and audio input.

        Args:
            instruction (str): A textual instruction guiding the model's behavior.
            audio (np.ndarray or torch.Tensor): The input audio waveform.
            sr (int): Sampling rate of the audio.
            max_new_tokens (int): Maximum number of tokens to generate. Default is 1024.

        Returns:
            response (str): The generated response based on the instruction and audio input.
        """
        return response
```

### 3. Register Your Model

Add your model to `src/models/__init__.py`:

```python
from src.models.mymodel import MyModel

models_map = {
    ...,
    "my_model": MyModel,
}
```

## Example Implementation

Here's a minimal example of integrating a new model:

```python
import torch
from src.models.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_my_model(model_path)  # Your model loading logic
        self.processor = load_my_processor()    # Your processor loading logic
        
    def chat_mode(self, audio, sr, max_new_tokens=2048):
        # Process audio
        processed_audio = self.processor(audio, sr)
        
        # Generate response
        response = self.model.generate(processed_audio)
        return response
        
    def prompt_mode(self, instruction, audio, sr, max_new_tokens=2048):
        # Process audio and instruction
        processed_input = self.processor(instruction, audio, sr)
        
        # Generate response
        response = self.model.generate(processed_input)
        return response
```

## Testing Your Integration

1. Test your model with the provided evaluation scripts:
```shell
python generate.py --model my_model --model_path /path/to/model --task asr
```

2. Verify results with the evaluation script:
```shell
python evaluate.py --model my_model --result_path results/my_model_results
```

## Support

For questions or issues about model integration, please:
1. Check existing model implementations in `src/models/` for reference
2. Create an issue in the repository with detailed information
3. Follow the code style of existing implementations
