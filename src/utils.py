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
    


def load_result_files(intput_dir):
    """
    Load result files from the input directory.
    """
    result_files = []
    if os.path.isfile(intput_dir):
        result_files.append(intput_dir)
    else:
        for root, dirs, files in os.walk(intput_dir):
            for file in files:
                if file.endswith('.json'):
                    result_files.append(os.path.join(root, file))
    return result_files

