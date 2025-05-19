# Model Integration Guide

This guide explains how to integrate your own models into the APEX-SUPERB framework. Our framework supports various types of Audio Language Models (ALLMs) and provides a flexible interface for adding new ones.

## Basic Structure

All models in APEX-SUPERB inherit from the `BaseModel` class, which defines three core interaction modes:

1. `chat_mode`: For open-ended conversations with audio input
2. `prompt_mode`: For task-specific interactions with audio input and instructions
3. `text_mode`: For text-only interactions

## Steps to Add Your Model

### 1. Create a New Model File

Create a new Python file in the `src/models` directory with your model name (e.g., `my_model.py`).

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
        # Implement open-ended conversation with audio input
        # audio: audio waveform
        # sr: sampling rate (usually 16000)
        return response
        
    def prompt_mode(self, instruction, audio, sr, max_new_tokens=2048):
        # Implement task-specific interaction
        # instruction: task-specific prompt
        # audio: audio waveform
        # sr: sampling rate
        return response
        
    def text_mode(self, instruction, text, max_new_tokens=2048):
        # Implement text-only interaction
        # Optional: implement if your model supports text-only mode
        return response
```

### 3. Register Your Model

Add your model to `src/models/__init__.py`:

```python
from .my_model import MyModel

__all__ = [
    ...,
    "MyModel",
]
```

## Implementation Guidelines

1. **Audio Processing**:
   - Input audio is typically provided as a waveform with a 16kHz sampling rate
   - Consider implementing audio preprocessing if your model requires specific formats

2. **Device Management**:
   - Always handle GPU/CPU device placement appropriately
   - Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

3. **Memory Efficiency**:
   - Consider implementing memory-efficient techniques for large models
   - Use `torch.cuda.empty_cache()` when appropriate

4. **Error Handling**:
   - Implement proper error handling for audio processing and model inference
   - Validate input formats and parameters

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
python evaluate.py --model my_model --result_path results/my_model_results --task asr
```

## Common Issues and Solutions

1. **Memory Management**:
   - If encountering OOM errors, consider implementing gradient checkpointing or model sharding
   - Use `torch.cuda.empty_cache()` between heavy operations

2. **Speed Optimization**:
   - Consider implementing batched processing if applicable
   - Use `@torch.no_grad()` for inference
   - Implement caching mechanisms if appropriate

3. **Input Validation**:
   - Always validate audio sampling rate (should be 16kHz)
   - Check audio duration and implement chunking if necessary

## Support

For questions or issues about model integration, please:
1. Check existing model implementations in `src/models/` for reference
2. Create an issue in the repository with detailed information
3. Follow the code style of existing implementations
