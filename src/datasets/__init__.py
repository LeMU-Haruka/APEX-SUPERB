from .base_dataset import TaskDataset, ChoiceDataset

datasets_map = {
    'mmlu': ChoiceDataset,
    'AlpacaEval': TaskDataset,
}


DATASET_MAP = {
    'AlpacaEval': '/userhome/datasets/AlpacaEval_cosyvoice',
    'mmlu': '/userhome/datasets/mmlu',
    'nutshell': '/userhome/datasets/nutshell',
}