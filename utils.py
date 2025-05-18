import json
import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed to {seed}")
    


def load_result_files(path):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))
    print(f"Found {len(files)} files in {path}")
    return files


def extract_json(text):
    """
    Finds the first valid JSON object or array substring in a larger string.

    Handles basic nesting of braces and brackets.
    Does NOT perfectly handle braces/brackets inside JSON strings.
    """
    potential_starts = [i for i, char in enumerate(text) if char in '{[']

    for start_index in potential_starts:
        brace_level = 0
        bracket_level = 0
        potential_end = -1
        start_char = text[start_index]

        for i in range(start_index, len(text)):
            char = text[i]

            # Basic handling: doesn't account for quotes yet
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            elif char == '[':
                bracket_level += 1
            elif char == ']':
                bracket_level -= 1

            # Check if we've closed the initial brace/bracket
            if start_char == '{' and brace_level == 0 and bracket_level == 0 and i > start_index:
                potential_end = i
                break
            elif start_char == '[' and bracket_level == 0 and brace_level == 0 and i > start_index:
                potential_end = i
                break
            # Added safety: stop if counts go negative (means mismatch)
            elif brace_level < 0 or bracket_level < 0:
                break


        if potential_end != -1:
            potential_json = text[start_index : potential_end + 1]
            try:
                # The crucial step: Validate by trying to parse
                json.loads(potential_json)
                return potential_json # Return the first valid one found
            except json.JSONDecodeError:
                # It looked like JSON but wasn't valid, continue searching
                continue

    return None # No valid JSON found